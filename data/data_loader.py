# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/08/24 17:25:00
@Author  :   Li Zhishuai
@Contact :   lizhishuai@sensetime.com
@Desc    :   None
'''
import os, sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))


class GeneDataset():

    def __init__(
        self,
        lags=24,
        split_rate=0.5,
    ):
        self.lags = lags
        self.split_rate = split_rate
        self.soruce_file = os.path.join(os.path.dirname(__file__), "ORI_DATA")
        self.save_file = os.path.join(os.path.dirname(__file__), "PRO_DATA")

    def data_process(self, dataset):
        if not os.path.exists(os.path.join(self.soruce_file, f'{dataset}/dataset.npz')):
            ret = np.load(os.path.join(self.soruce_file, \
                f'{dataset}/dataset.npz'), allow_pickle=True)
            return ret['train'], ret['test'], ret['adj'], 0
        else:
            adj = get_adj(dataset)
            nodes_split = np.load(self.soruce_file + 
                                    f"/{dataset}/nodes_split.npz", allow_pickle=True)
            data = pd.read_csv(
                    os.path.join(self.soruce_file, f"{dataset}/{dataset}_data.csv"),
                    index_col=0)
            data = data.iloc[:288 * 30 * 2]
            # if dataset == "pems":
            #     self.split_rate = 0.5
            # else:
            #     data = data.iloc[:288 * 30 * 2]
            use_nodes = sorted(
                list(nodes_split['trn_nodes']) +
                list(nodes_split['tst_nodes']))
            data = data.iloc[:, use_nodes]

            ## cal the scaler from train data
            scaler = MinMaxScaler().fit(data) 
            # ## scale all the data
            data = scaler.transform(data)
            data = np.lib.stride_tricks.sliding_window_view(
                data, (self.lags), axis=0)[::self.lags]

            self.road_num = data.shape[1]
            split_pnt = int(self.split_rate * data.shape[0])
            train, test = data[:split_pnt], data[split_pnt:]
            if dataset == "pems":
                test, train = train, test
            train, test = np.transpose(train, (0, 2, 1)), np.transpose(
                test, (0, 2, 1))
            if dataset == "nerl":
                ## add noise
                test = test.reshape(-1, self.road_num)
                np.random.seed(0)
                know_data_mask = np.random.uniform(0, 1, test.flatten().shape[0]) > 0.03
                know_data_mask = know_data_mask.reshape(test.shape)
                know_data_mask[:, nodes_split['test_nodes_id']] = 1
                test = test * know_data_mask
                test = test.reshape(-1, self.lags, self.road_num)
            np.savez(os.path.join(self.soruce_file, f'{dataset}/dataset.npz'),
                        train=train,
                        test=test,
                        adj=adj,
                        data_min=scaler.data_min_,
                        data_max=scaler.data_max_)
            print("Reprepare .npz done!")
            return train, test, adj, scaler


class TimeDataset(Dataset):

    def __init__(self, hops=3):
        super().__init__()
        self.hops = hops

    def len(self):
        return NotImplementedError

    def get(self, idx):
        return NotImplementedError


class TrainDataset(TimeDataset):

    def __init__(self, data, dataset, rep_num):
        super().__init__()
        ## enlarge the dataset by repeating 'rep_num' times for various aug
        # if dataset == "pems":
        #     rep_num = 3
        self.data = torch.from_numpy(np.tile(data.copy(), [rep_num, 1, 1])).float()
        self.adj = torch.from_numpy(get_adj(dataset)).float()
        if dataset == "pems":
            self.adj = self.adj + self.adj.t()
            self.adj = torch.where(self.adj > 1, self.adj / 2, self.adj)
        nodes_split = np.load(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
            f'/{dataset}/nodes_split.npz', allow_pickle=True)
        self.test_nodes = torch.from_numpy(nodes_split['test_nodes_id'])
        self.know_nodes = nodes_split['train_nodes_id'].tolist()
        ## num_samples * input_len * all_nodes
        self.data_len = self.data.shape[0]
        self.num_aug_nodes = int(self.data.shape[-1] * 0.2)

    def len(self, ):
        return self.data_len

    def get(self, idx):
        ready_nodes = sorted(self.know_nodes)
        np.random.seed(idx)
        map_id = np.random.choice(np.arange(len(ready_nodes)),
                                  self.num_aug_nodes, replace=False)
        idx_of_train_node = np.array(ready_nodes)[map_id]
        adj = self.adj[ready_nodes, :][:, ready_nodes]
        ## get known nodes for training
        data = self.data[idx, :, ready_nodes].transpose(0, 1)
        node_mask = torch.ones_like(data)
        node_mask[map_id] = 0
        edge_index = adj.nonzero().t().contiguous()
        edge_weight = adj[edge_index[0], edge_index[1]]
        adj_t = adj.t()
        edge_index2 = adj_t.nonzero().t().contiguous()
        edge_weight2 = adj_t[edge_index2[0], edge_index2[1]]
        ## topo aug
        nodes_centrality = (adj[map_id, :] > 0).sum(1)
        mean_c = (adj > 0).sum(1).float().mean().int()
        p = (nodes_centrality - mean_c) / torch.max((adj > 0).sum(1))
        p[p < 0] = 0.
        adj_corrupt = adj.clone().fill_diagonal_(0)
        for i, item in enumerate(map_id):
            if p[i] == 0:
                continue
            num_smps = (nodes_centrality - mean_c)[i]
            smps = torch.arange(adj.shape[0])[adj[item, :] > 0]
            unif = torch.ones(smps.shape[0])
            idx = unif.multinomial(num_smps)
            smp_idx = torch.bernoulli(p[i].repeat(smps[idx].shape[0])).bool()
            adj_corrupt[item, smps[idx][smp_idx]] = 0
        adj_corrupt = adj_corrupt.fill_diagonal_(1)
        crpt_edge_index = adj_corrupt.nonzero().t().contiguous()
        crpt_edge_weight = adj_corrupt[crpt_edge_index[0], crpt_edge_index[1]]
        adj_corrupt_t = adj_corrupt.t()
        crpt_edge_index2 = adj_corrupt_t.nonzero().t().contiguous()
        crpt_edge_weight2 = adj_corrupt_t[crpt_edge_index2[0], crpt_edge_index2[1]]
        return Data(x=node_mask * data, y=data, edge_index=edge_index, \
            edge_weight=edge_weight, edge_index2=edge_index2, \
                edge_weight2=edge_weight2, \
                    node_id=idx_of_train_node, idx_of_node=data.shape[0],\
                        map_id=map_id, crpt_edge_index=crpt_edge_index, crpt_edge_weight=crpt_edge_weight, \
                            crpt_edge_index2=crpt_edge_index2, crpt_edge_weight2=crpt_edge_weight2)


class TestDataset(TimeDataset):

    def __init__(self, data, dataset):
        super().__init__()
        self.data = torch.from_numpy(data.copy()).float()
        self.adj = torch.from_numpy(get_adj(dataset)).float()
        if dataset == "pems":
            self.adj = self.adj + self.adj.t()
            self.adj = torch.where(self.adj > 1, self.adj / 2, self.adj)
        nodes_split = np.load(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
            f'/{dataset}/nodes_split.npz', allow_pickle=True)
        self.test_nodes = nodes_split['test_nodes_id']
        self.know_nodes = nodes_split['train_nodes_id'].tolist()
        ## num_samples * input_len * all_nodes
        self.data_len = self.data.shape[0]

    def len(self, ):
        return self.data_len

    def get(self, idx):
        edge_index = self.adj.nonzero().t().contiguous()
        edge_weight = self.adj[edge_index[0], edge_index[1]]
        data = self.data[idx].transpose(0, 1)
        node_mask = torch.ones_like(data)
        node_mask[self.test_nodes] = 0
        adj_t = self.adj.t()
        edge_index2 = adj_t.nonzero().t().contiguous()
        edge_weight2 = adj_t[edge_index2[0], edge_index2[1]]
        return Data(x=(data * node_mask), y=data,\
                edge_index=edge_index, edge_weight=edge_weight, \
                    edge_index2=edge_index2, edge_weight2=edge_weight2, \
                    node_id=self.test_nodes, idx_of_node=data.shape[0], map_id=self.test_nodes)

def get_pems_adj():
    all_nodes = pd.read_csv(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
            '/pems/pems_data.csv',
                       index_col=0).columns
    A = pd.read_csv(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
            '/pems/pems_adj.csv', index_col=0).loc[\
            list(map(float, all_nodes)), list(all_nodes)]
    
    # print(A)
    distance_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
            '/pems/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'}, header=None)
    A.values[:] = np.inf#np.zeros((len(list(nodes_id)), len(list(nodes_id))))
    
    for row in distance_df.values:
        if not (row[0] in list(map(float, all_nodes)) and row[1] in list(map(float, all_nodes))):
            continue
        A.loc[float(row[0]), str(int(row[1]))] = row[2]
        # A.loc[float(row[1]), str(int(row[0]))] = row[2]
    dist_mx = A.values[:]
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    A.values[:] = np.exp(-np.square(dist_mx / std))
    A.index = np.arange(A.shape[0])
    A.columns = np.arange(A.shape[0])
    return A


def get_adj(dataset):
    if dataset == 'pems':
        A = get_pems_adj()
    else:
        A = pd.read_csv(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
                f'/{dataset}/{dataset}_adj.csv', index_col=0)
    A[A < 0.1] = 0
    loop_idx = sum(A.values > 0) > 3
    A_s = A.iloc[loop_idx, loop_idx]

    train_size = int(0.8 * A_s.shape[0])

    train_list = sorted(
        np.random.choice(A_s.shape[0], train_size, replace=False))
    test_list = sorted(list(set(range(A_s.shape[0])) - set(train_list)))

    train_nodes = []
    for node in train_list:
        if sum(A_s.iloc[node, train_list].values) > 2:
            train_nodes.append(node)
    test_nodes = []
    test_nodes_dict = {}
    for node in test_list:
        if sum(A_s.iloc[node, train_nodes].values) > 2:
            test_nodes.append(node)
            test_nodes_dict[node] = 1
    trn_tst = sorted(train_nodes + test_nodes)

    used_A = A_s.iloc[trn_tst, trn_tst]
    np.savez(os.path.join(os.path.dirname(__file__), "ORI_DATA") + \
            f'/{dataset}/nodes_split.npz', \
            trn_nodes=A_s.index[train_nodes], tst_nodes=A_s.index[test_nodes], \
            train_nodes_id=[idx for idx, val in enumerate(used_A.index) \
                if val in A_s.index[train_nodes]], \
            test_nodes_id=[idx for idx, val in enumerate(used_A.index) \
                if val in A_s.index[test_nodes]])
    return used_A.values


def data_prepare(dataset, rep_times):
    data = GeneDataset()
    train, test, _, _ = data.data_process(dataset)
    train_set = TrainDataset(train, dataset, rep_num=rep_times)
    test_set = TestDataset(test, dataset)
    train_loader = DataLoader(train_set)
    test_loader = DataLoader(test_set)

    datafile = os.path.join(os.path.dirname(__file__), f"PRO_DATA/{dataset}")
    if not os.path.exists(datafile):
        os.mkdir(datafile)

    ## train.data datapro
    data_list = []
    for j, data in enumerate(train_loader):
        data_list.append(data)
    torch.save(data_list, datafile + '/train.data')

    ## test.data datapro
    data_list = []
    for j, data in enumerate(test_loader):
        data_list.append(data)
    torch.save(data_list, datafile + '/test.data')

    print("Reprepare .data done!")


def data_loader(batch_size, dataset, rep_times=3):
    datafile = os.path.join(os.path.dirname(__file__), f"PRO_DATA/{dataset}")
    # if not os.path.exists(os.path.dirname(__file__) + f"/PRO_DATA/{dataset}/train.data"):
    data_prepare(dataset, rep_times=rep_times)
    data_list = torch.load(datafile + '/train.data')
    trainloader = DataLoader(data_list,
                             num_workers=1,
                             batch_size=batch_size,
                             shuffle=True)
    data_list = torch.load(datafile + '/test.data')
    testloader = DataLoader(data_list,
                            num_workers=1,
                            batch_size=batch_size * 10)
    return trainloader, testloader, 0


if __name__ == '__main__':
    # data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'node_300.csv'),
    #                    index_col=0)  #.drop(columns='401489')
    # # split_pnt = int(0.8 * data_num)
    # ## cal the scaler from train data
    # scaler = StandardScaler().fit(data)
    # # ## scale all the data
    # x = scaler.transform(data)

    # import matplotlib.pyplot as plt
    # for i in range(2):
    #     plt.plot(x[:288 * 7, i])
    #     # plt.plot(x[0].numpy()[:, 1])
    # plt.savefig("sensor_zscore.png")
    # data_prepare(dataset='pems', rep_times=1)
    data = GeneDataset()
    train, test, _, _ = data.data_process('ushcn')
    # get_adj()
