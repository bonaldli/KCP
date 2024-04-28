from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

torch.manual_seed(0)

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, configs):
        super(MLPDecoder, self).__init__()
        hid_dim = configs['dec_hid']
        self.model = nn.Sequential(nn.Linear(input_dim, hid_dim * 4), \
            nn.ReLU(), nn.Linear(hid_dim * 4, hid_dim * 2), nn.ReLU(), \
                nn.Linear(hid_dim * 2, configs["inp_len"]))

    def forward(self, x):
        return self.model(x)

class AdaAug(nn.Module):
    def __init__(self, seq_len=24, aug_nums=2):
        '''adaptive augmentation
        '''
        super(AdaAug, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(seq_len, 32), nn.ReLU(), \
            nn.Linear(32, aug_nums))

    def forward(self, input):
        bsxn = input.x.size(0)
        bs = len(input.node_id)
        num_nodes = bsxn // bs
        seq_len = input.x.size(-1)

        krig_idx = torch.from_numpy(np.array(input.map_id)).to(input.x.device).squeeze()
        smp_x = input.y.reshape(-1, num_nodes, seq_len).\
            gather(1, krig_idx[:, :, None].repeat(1, 1, seq_len)).reshape(-1, seq_len)
        # shuffle = torch.randperm(seq_len).to(input.x.device)
        mask = torch.FloatTensor(smp_x.size()).to(input.x.device).uniform_() > 0.1
        data_aug = torch.cat([torch.zeros_like(smp_x, device=input.x.device).unsqueeze(2), \
            (mask * smp_x).unsqueeze(2)], dim=-1)

        krig_feat = input.y.reshape(-1, num_nodes, seq_len).\
            gather(1, krig_idx[:, :, None].repeat(1, 1, seq_len)).reshape(-1, seq_len)
        
        out = self.mlp(krig_feat)
        gb_smp = F.gumbel_softmax(out, tau=0.1, hard=True)
        cusum = torch.cat([torch.tensor([0], device=input.x.device), \
                    input.idx_of_node], 0)[:-1]
        krig_idx = torch.cumsum(cusum, 0).unsqueeze(1).repeat(1, input.map_id[0][0].shape[0]\
            ).view(-1,) + krig_idx.view(-1,)
        input.x[krig_idx] = (gb_smp * data_aug.transpose(0, 1).detach()).sum(-1).transpose(0, 1)        
        return input


class Encoder(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.rep_dim = configs['emb_size']
        self.st_blocks = nn.ModuleList([SAGEConv(configs['inp_len'], configs["dec_hid"], project=True), \
                SAGEConv(configs["dec_hid"], self.rep_dim, project=True)])  
        self.norm = nn.ModuleList([nn.LayerNorm(configs["dec_hid"]), nn.LayerNorm(self.rep_dim)])
        self.lyn = nn.LayerNorm(self.rep_dim)
        self.lin = nn.Linear(self.rep_dim, self.rep_dim)
 
    def forward(self, x, input, aug_view=False):
        bsxn = x.size(0)
        bs = len(input.node_id)
        num_nodes = bsxn // bs
        for idx, st_block in enumerate(self.st_blocks):
            if aug_view:
                x1 = st_block(x, input.crpt_edge_index)  
            else:
                x1 = st_block(x, input.edge_index)
            # x = self.norm[idx](x1)
            x = F.relu(x1)
        x = x.reshape(bsxn, -1)
        # x = self.lyn(self.lin(x))
        x = self.lin(x) 
        krig_idx = torch.from_numpy(np.array(input.map_id)).to(x.device).squeeze()
        local_feat = x.reshape(-1, num_nodes, self.rep_dim).\
            gather(1, krig_idx[:, :, None].repeat(1, 1, self.rep_dim)).reshape(-1, self.rep_dim)
        return local_feat, x


class SelfGNN(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.ada_aug = AdaAug()
        self.student_encoder = Encoder(configs)
        self.rep_dim = self.student_encoder.rep_dim
        self.out_decoder = MLPDecoder(self.rep_dim, configs)
        self.mseloss = nn.L1Loss()
        self.prototype_head = PrototypeHead(c_in=self.rep_dim, nmb_prototype=configs['nmb_p'])
        self.neighbor_contrast = NeigborContrast(self.rep_dim)

    def encode(self, x, input, aug_view=False):
        encoder = self.student_encoder
        return encoder(x, input, aug_view)

    def decoder_trainer(self, x, input):
        bsxn = input.x.size(0)
        bs = len(input.node_id)
        seq_len = input.x.size(-1)
        num_nodes = bsxn // bs
        krig_idx = torch.from_numpy(np.array(input.map_id)).to(x.device).squeeze()
        y = input.y.reshape(-1, num_nodes, seq_len).gather(1, krig_idx[:, :, None].repeat(1, 1, seq_len)).reshape(-1, seq_len)
        return self.mseloss(x, input.y)

    def forward(self, input):
        # input = self.ada_aug(input)
        bsxn = input.x.size(0)
        bs = len(input.node_id)
        num_nodes = bsxn // bs

        x1, x2 = input.y, input.x
        _, x1 = self.encode(x1, input)
        v2_local, x2 = self.encode(x2, input, aug_view=False)

        loss1 = self.prototype_head(x1, x2)
        loss2, acc = self.neighbor_contrast(x1.reshape(-1, num_nodes, self.rep_dim), \
            x2.reshape(-1, num_nodes, self.rep_dim), input)
        loss = loss2 + loss1
        return loss.mean(), acc


class PrototypeHead(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, nmb_prototype, tau=0.3):
        super(PrototypeHead, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        
        self.tau = tau
        self.d_model = c_in

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)
        
        # l2norm avoids nan of Q in sinkhorn
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model))) # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model))) # nd -> nk
        with torch.no_grad():
            q1 = distributed_sinkhorn(zc1.detach())
            q2 = distributed_sinkhorn(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2

@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, sinkhorn_iterations=20):
    """Apply the distributed sinknorn optimization on the scores matrix to find
    the assignments."""
    eps_num_stab = 1e-12
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        u = torch.sum(Q, dim=1, keepdim=True)
        if len(torch.nonzero(u == 0)) > 0:
            Q += eps_num_stab
            u = torch.sum(Q, dim=1, keepdim=True, dtype=Q.dtype)
        Q /= u
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()


class NeigborContrast(nn.Module):

    def __init__(self, c_in):
        super(NeigborContrast, self).__init__()
        self.topk = 5
        self.sa = nn.Linear(c_in, 1)
        self.lin = nn.Linear(c_in, c_in)
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

    def ntx(self, z1, z2, pos_mask, tau=0.5, normalize=True):
        if normalize:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
        sim_mat = torch.bmm(z1, z2.transpose(1, 2))
        sim_mat = torch.exp(sim_mat / tau)
        return -torch.log((sim_mat * pos_mask).sum(-1) / sim_mat.sum(-1) / pos_mask.sum(-1)).mean(), 0

    def neighbor_readout(self, z1, z2, input):
        bs, num_nodes, rep_dim = z1.size()
        adj = torch.zeros([num_nodes, num_nodes]).to(z1.device)
        edge_idx = torch.where(input.edge_index[0] < num_nodes)[0][-1] + 1
        adj[input.edge_index[0][:edge_idx], input.edge_index[1][:edge_idx]] = input.edge_weight[:edge_idx]
        adj = adj.fill_diagonal_(1)
        adj_idx = adj > 0
        all_negbs_emb = torch.empty_like(z1).to(z1.device)
        topk = self.topk
        for i in range(num_nodes):
            negbor_weight = adj[i][adj_idx[i]] / adj[i][adj_idx[i]].sum()
            topk_w, topk_idx = torch.sort(negbor_weight, descending=True)
            nbs_feats = z1[:, adj_idx[i]][:, topk_idx[:topk]]
            negbor_weight = F.softmax(self.sa(nbs_feats).squeeze(2), dim=-1)
            ans = self.lin((nbs_feats * negbor_weight.unsqueeze(2)).sum(1))
            all_negbs_emb[:, i] = ans
        bs_idx = torch.randperm(bs)
        node_idx = torch.randperm(num_nodes)
        shuf_negbs_emb = all_negbs_emb[bs_idx][:, node_idx]
        return z2, all_negbs_emb, shuf_negbs_emb

    def forward(self, z1, z2, input):
        bs, num_nodes, rep_dim = z1.size()
        lbl_rl = torch.ones((bs, num_nodes))
        lbl_fk = torch.zeros((bs, num_nodes))
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1).to(z1.device)    
        s, h, shuf_h = self.neighbor_readout(z1, z2, input)
        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, lbl)
        acc = ((torch.sigmoid(logits) > 0.5).int() == lbl).sum() / torch.ones_like(lbl, device=lbl.device).sum()
        return loss, acc.item()

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, h_rl, h_fk):
        # score of real and fake, (batch_size, num_nodes)
        h_rl = F.normalize(h_rl, p=2, dim=-1)
        h_fk = F.normalize(h_fk, p=2, dim=-1)
        s = F.normalize(s, p=2, dim=-1)

        sc_rl = torch.diagonal(torch.bmm(s, h_rl.transpose(1, 2)), dim1=-2, dim2=-1)
        sc_fk = torch.diagonal(torch.bmm(s, h_fk.transpose(1, 2)), dim1=-2, dim2=-1)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits