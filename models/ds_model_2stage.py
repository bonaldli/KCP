import os, sys
import time
import yaml
import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.data_loader import data_loader
from models.model import SelfGNN
from utils import adjust_learning_rate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

config_filename = f'configs/cfg.yaml'
configs = yaml.load(open(config_filename), Loader=yaml.FullLoader)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=configs['dataset'][configs['dataset_idx']], 
                type=str, help='the configuration to use')
parser.add_argument('--seed', default=configs['seed'], 
                type=str, help='the configuration to use')
args = parser.parse_args()

set_seed(int(args.seed))

tbpath = os.path.dirname(__file__) + f'/tblog/waa{time.time()}'
os.makedirs(tbpath)
writer = SummaryWriter(tbpath)

configs['downtask_lr'] = float(args.lr)

class MLPDec(nn.Module):
    """MLP decoder for downstream task 
    """

    def __init__(self, input_dim, hid_dim, configs):
        super(MLPDec, self).__init__()
        self.up_enc = SelfGNN(configs)
        self.model = nn.Sequential(nn.Linear(input_dim, hid_dim * 2), \
            nn.ReLU(), nn.Linear(hid_dim * 2, configs["inp_len"]))

    def forward(self, x, input):
        local_feat, output = self.up_enc.student_encoder(x, input)
        return self.model(local_feat)


def test_loss(ds_model, test_loader, net):
    pred = []
    gtrh = []
    global max_r2
    global max_metrics
    ds_model.eval()
    num_nodes = len(nodes_split['trn_nodes']) + len(nodes_split['tst_nodes'])
    for idx, data in enumerate(test_loader):
        data = data.to(device)
        krig_idx = torch.from_numpy(np.array(
            data.map_id)).to(device).squeeze()
        test_set = data.x.view(-1, num_nodes, configs["inp_len"]).permute(0, 2, 1).reshape(-1, num_nodes).cpu().numpy()
        # np.random.seed(0)
        # know_data_mask = np.random.uniform(0, 1, test_set.flatten().shape[0]) > 0.1
        # know_data_mask = know_data_mask.reshape(test_set.shape)
        # know_data_mask[:, nodes_split['test_nodes_id']] = 1
        know_data_mask = np.ones_like(test_set)
        data.x = torch.from_numpy(test_set * know_data_mask).to(device).reshape(-1, \
                configs["inp_len"], num_nodes).transpose(1, 2).reshape(-1, configs["inp_len"])
        y = data.y.reshape(-1, num_nodes, configs["inp_len"]).\
            gather(1, krig_idx[:, :, None].repeat(1, 1, configs["inp_len"])).\
                reshape(-1, configs["inp_len"])
        ret_local = ds_model(data.x, data).reshape(-1, configs["inp_len"])
        pred.append(ret_local.cpu().detach().numpy())
        gtrh.append(y.cpu().detach().numpy())

    pred = np.concatenate(pred,
                            0).reshape(-1, nodes_split['tst_nodes'].shape[0],
                                        configs["inp_len"])
    pred = pred.transpose(1, 0, 2).reshape(
        nodes_split['tst_nodes'].shape[0], -1) ## test_nodes * timestamps
    gtrh = np.concatenate(gtrh,
                            0).reshape(-1, nodes_split['tst_nodes'].shape[0],
                                        configs["inp_len"])
    gtrh = gtrh.transpose(1, 0, 2).reshape(
        nodes_split['tst_nodes'].shape[0], -1) ## test_nodes * timestamps

    data = np.load("data/ORI_DATA" + f"/{args.dataset}/dataset.npz", allow_pickle=True)
    max_data = data['data_max'][nodes_split['test_nodes_id']]
    min_data = data['data_min'][nodes_split['test_nodes_id']]
    pred = (pred.transpose(1, 0) * (max_data - min_data) +
            min_data).transpose(1, 0)
    gtrh = (gtrh.transpose(1, 0) * (max_data - min_data) +
            min_data).transpose(1, 0)

    def mape(pred, gtrh):
        gtrh = gtrh.copy().reshape(-1, )
        pred = pred.copy().reshape(-1, )
        gtrh[gtrh < 10] = np.nan
        return np.nanmean(np.abs(pred - gtrh) / gtrh)

    if args.dataset == 'wind':
        pred = pred[1]
        gtrh = gtrh[1]
    pred[gtrh == 0] = 0
    test_mask = np.ones(gtrh.shape, dtype=bool)
    test_mask[gtrh == 0] = 0

    mask = np.ones(gtrh.shape, dtype=bool)
    MAE = np.sum(
        np.abs(pred[mask].reshape(-1, ) -
                gtrh[mask].reshape(-1, ))) / test_mask.sum()
    RMSE = np.sqrt(
        np.sum((pred[mask].reshape(-1, ) - gtrh[mask].reshape(-1, ))**2) /
        test_mask.sum())
    MAPE = mape(pred[mask], gtrh[mask])
    r2 = r2_score(gtrh[mask].reshape(-1, ), pred[mask].reshape(-1, ))
    print(f'{MAE:.3f}, {RMSE:.3f}, {MAPE:.3f}, {r2:.3f}')
    print("test size: ", gtrh.shape)
    print("train size: ", len(nodes_split['trn_nodes']))
    if r2 > max_r2:
        max_metrics = [MAE, RMSE, MAPE, r2]
        max_r2 = r2


def downstream_train(selfgnn, train_data, test_loader):
    mseloss = nn.L1Loss()
    ds_model = MLPDec(*configs["downtask_layers"], configs).to(device)
    ds_model.up_enc.load_state_dict(selfgnn.state_dict())

    opt = torch.optim.Adam([{
        "params": ds_model.model.parameters(),
        'lr': configs['downtask_lr']
        }, {
        "params": ds_model.up_enc.student_encoder.parameters(),
        "lr": configs["finetune_lr"]
    }])
    ds_model.to(device)
    ds_model.train()
    early_stop = 30
    flag_loss = 1e10
    num_trn_nodes = len(nodes_split['train_nodes_id'])
    print("*" * 10, "Start downtask training", "*" * 10)
    for e in range(configs["downtask_epochs"]):
        loss_e = 0
        for i, data in enumerate(train_data):
            data = data.to(device)
            out = ds_model(data.x, data)
            krig_idx = torch.from_numpy(np.array(data.map_id)).to(device).squeeze()
            y = data.y.reshape(-1, num_trn_nodes, configs['inp_len']).gather(1, \
                krig_idx[:, :, None].repeat(1, 1, configs['inp_len'])).reshape(-1, configs['inp_len'])
            loss = mseloss(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_e += loss.item()
        if loss_e / (i + 1) < flag_loss:
            patience = 0
            flag_loss = loss_e / (i + 1)
        patience += 1
        if early_stop < patience:
            break
        if e > 50:
            with torch.no_grad():
                test_loss(ds_model, test_loader, selfgnn)
        writer.add_scalar("loss/Train Loss", loss_e / (i + 1), e)
        writer.add_scalar("loss/Test MAE", max_metrics[0], e)
    #     print(f'Epoch {e},\t kriging loss: {loss_e / (i + 1):.6f}')
    print("*" * 10, "Training finish!", "*" * 10)
    return ds_model

if __name__ == '__main__':

    nodes_split = np.load("data/ORI_DATA" + f'/{args.dataset}/nodes_split.npz', allow_pickle=True)
    device = torch.device(configs["tst_device"]) if torch.cuda.is_available() else \
        torch.device('cpu')
    train_loader, test_loader, scaler = data_loader(configs['batch_size'], args.dataset)
    net = SelfGNN(configs)
    net.to(device)
    save_path = f'model_save/{args.dataset}'
    net.load_state_dict(torch.load(os.path.join(save_path, f"best_model_{args.dataset}.pth"))['net_state_dict'])
    global max_r2s
    max_r2 = 0
    global max_metrics
    max_metrics = [0, 0, 0, 0]
    net.eval()
    ds_m = "MLP"

    ds_model = downstream_train(net, train_loader, test_loader)
    print("Best metrics: ", max_metrics)

