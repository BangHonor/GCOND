import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import gc
from torch_geometric.loader import *
import deeprobust.graph.utils as utils
from torch_sparse import SparseTensor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)#this py file's dir + data + name
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag']:
        dataset = PygNodePropPredDataset(name=name,
                                    transform=T.ToSparseTensor())
        print("pyg_data:",dataset[0])#pyg_dataset: Data(edge_index=[2, 1166243], node_year=[169343, 1], x=[169343, 128], y=[169343, 1])
        split_index = dataset.get_idx_split()
        print("split_index:",(split_index))#{'train': tensor([     0,      1,      2,  ..., 169145, 169148, 169251]), 'valid': tensor([   349,    357,    366,  ..., 169185, 169261, 169296]), 'test': tensor([   346,    398,    451,  ..., 169340, 169341, 169342])}
    else:
        raise NotImplementedError

    # x = dataset[0].x.numpy()
    # y = dataset[0].y.numpy().reshape(-1)
    # n=y.max()+1
    # x_tsne = TSNE(n_components=2).fit_transform(x)

    # for i in range(n):
    #     plt.figure(figsize=(8,8))
    #     plt.scatter(x_tsne[y == i, 0], x_tsne[y == i, 1])
    #     plt.savefig('/home/xzb/GCond/pictures/'+name+str(i)+'.png')

    if transform is not None and normalize_features:#transform pyg_data
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:#归一
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dpr_data = Pyg2Dpr(dataset)
    if name in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag']:
        # the features are different from the features provided by GraphSAINT
        # normalize features, following graphsaint
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data


class Pyg2Dpr(Dataset):#input dataset and get the divided one. if we input partitioned dataset, then we can get what we want
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag']: # symmetrization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pyg_data.edge_index = to_undirected(edge_index=pyg_data.edge_index, edge_attr=None,num_nodes=pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),#化为普通的稀疏矩阵
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)
        print("train val test的长度:",len(self.idx_train),len(self.idx_val),len(self.idx_test))

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

class Transd2Ind:#based onn dpr_data,get feat,adj,label
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio):
        idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels#这里的adj是普通的稀疏矩阵
        self.nclass = int(labels.max()+1)

        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=labels[idx_train])

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]

        print('size of adj_train:', self.adj_train.shape)
        print('edges in adj_train:', self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=512, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        else:
            sizes = [15, 10, 5, 5, 5]
        
        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                if len(self.class_dict2[i])==0:#要考虑到有些分类是没有的
                    self.samplers.append(None)
                    continue
                node_idx = torch.LongTensor(self.class_dict2[i])#该类的index，可能大于256.此时就会采样多个batch
                self.samplers.append(NeighborSampler(adj,#返回的是一个batch的loader，里面可能有很多个batch
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=12, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]#对固定点进行采样,当长度没有256时取原长
        # batch = np.random.permutation(self.class_dict2[c])
        out = self.samplers[c].sample(batch)
        return out

    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(NeighborSampler(adj,
                                        node_idx=node_idx,
                                        sizes=sizes, batch_size=num,
                                        num_workers=12, return_e_id=False,
                                        num_nodes=adj.size(0),
                                        shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers-1][c].sample(batch)
        return out

class Transd2Ind_Large:
    # transductive setting to inductive setting

    def __init__(self, name, keep_ratio):
        dataset = PygNodePropPredDataset(name=name,
                                            transform=T.ToSparseTensor())#data数据一般都是tensor形式

        dataset.transform = T.NormalizeFeatures()#并没有实现归一化
        splits = dataset.get_idx_split()
        pyg_data=dataset[0]
        pyg_data.edge_index = to_undirected(edge_index=pyg_data.edge_index, num_nodes=pyg_data.num_nodes)
        n = pyg_data.num_nodes
        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),#化为普通的稀疏矩阵
                    (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        scaler = StandardScaler()
        scaler.fit(self.features[splits['train']])
        self.features = scaler.transform(self.features)

        # self.train_loader = NeighborLoader(
        #     pyg_data,
        #     # Sample 30 neighbors for each node for 2 iterations
        #     num_neighbors=[10,5],
        #     # Use a batch size of 128 for sampling training nodes
        #     shuffle=False,
        #     batch_size=10000,
        #     input_nodes=splits['train'],
        # )

        adj= utils.to_tensor(self.adj, None, device='cpu')
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)#D^(-0.5)*(A+I)*D^0.5
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm#正则化之后的邻接矩阵A 2708*2708
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()#所有东西丢入深度学习网络之前都需要转变为tensor

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)

        #太大的话省掉full
        # self.adj_full, self.feat_full, self.labels_full = self.adj, self.features, self.labels
        self.nclass = self.labels.max()+1
        self.idx_train = np.array(self.idx_train)
        self.idx_val = np.array(self.idx_val)
        self.idx_test = np.array(self.idx_test)

        # self.adj.save_npz('.npz', '/home/xzb/GCond/dataset/'+name+'/adj.npz')
        # np.save(self.adj, f'/home/xzb/GCond/dataset/'+name+'/adj.npz')
        # np.save(self.labels, f'/home/xzb/GCond/dataset/'+name+'/labels.npz')
        # np.save(self.features, f'/home/xzb/GCond/dataset/'+name+'/transform_features.npz')
        # torch.save(adj, f'/home/xzb/GCond/dataset/'+name+'/SparseTensor_adj.pt')
        # np.save(self.idx_train, f'/home/xzb/GCond/dataset/'+name+'/idx_train.npz')
        # np.save(self.idx_val, f'/home/xzb/GCond/dataset/'+name+'/idx_val.npz')
        # np.save(self.idx_test, f'/home/xzb/GCond/dataset/'+name+'/idx_test.npz')

        # self.adj=sp.load_npz('/home/xzb/GCond/dataset/'+name+'/adj.npz')
        # self.labels=np.load(f'/home/xzb/GCond/dataset/'+name+'/labels.npz')
        # self.features=np.load(f'/home/xzb/GCond/dataset/'+name+'/transform_features.npz')
        # adj=torch.load(f'/home/xzb/GCond/dataset/'+name+'/SparseTensor_adj.pt')
        # self.idx_train=np.load(f'/home/xzb/GCond/dataset/'+name+'/idx_train.npz')
        # self.idx_val=np.load(f'/home/xzb/GCond/dataset/'+name+'/idx_val.npz')
        # self.idx_test=np.load(f'/home/xzb/GCond/dataset/'+name+'/idx_test.npz')

        self.train_loader=NeighborSampler(adj,#返回的是一个batch的loader，里面可能有很多个batch
            node_idx=splits['train'],
            sizes=[-1,-1], 
            batch_size=4096,
            num_workers=12, 
            return_e_id=False,
            num_nodes=n,
            shuffle=False
        )

        if keep_ratio < 1:
            idx_train, _ = train_test_split(self.idx_train,
                                            random_state=None,
                                            train_size=keep_ratio,
                                            test_size=1-keep_ratio,
                                            stratify=self.labels[self.idx_train])
        
        self.adj_train = self.adj[np.ix_(self.idx_train, self.idx_train)]
        self.adj_val = self.adj[np.ix_(self.idx_val, self.idx_val)]
        self.adj_test = self.adj[np.ix_(self.idx_test, self.idx_test)]

        print('size of adj_train:', self.adj_train.shape)
        print('edges in adj_train:', self.adj_train.sum())

        self.labels_train = self.labels[self.idx_train]
        self.labels_val = self.labels[self.idx_val]
        self.labels_test = self.labels[self.idx_test]

        self.feat_train = self.features[self.idx_train]
        self.feat_val = self.features[self.idx_val]
        self.feat_test = self.features[self.idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]


# class Transd2Ind_partition:
#     # transductive setting to inductive setting

#     def __init__(self,features,adj,labels,idx_train,idx_test,idx_val, keep_ratio):
#         #把全部的feature adj label和index传进来
#         self.nclass = labels.max()+1
#         self.adj_full, self.feat_full, self.labels_full = adj, features, labels
#         self.idx_train = np.array(idx_train)
#         self.idx_val = np.array(idx_val)
#         self.idx_test = np.array(idx_test)


#         if keep_ratio < 1:
#             idx_train, _ = train_test_split(idx_train,
#                                             random_state=None,
#                                             train_size=keep_ratio,
#                                             test_size=1-keep_ratio,
#                                             stratify=labels[idx_train])

#         self.adj_train = adj[np.ix_(idx_train, idx_train)]
#         self.adj_val = adj[np.ix_(idx_val, idx_val)]
#         self.adj_test = adj[np.ix_(idx_test, idx_test)]

#         self.labels_train = labels[idx_train]
#         self.labels_val = labels[idx_val]
#         self.labels_test = labels[idx_test]

#         self.feat_train = features[idx_train]
#         self.feat_val = features[idx_val]
#         self.feat_test = features[idx_test]

#         self.class_dict = None
#         self.samplers = None
#         self.class_dict2 = None

#     def retrieve_class(self, c, num=256):
#         if self.class_dict is None:
#             self.class_dict = {}
#             for i in range(self.nclass):
#                 self.class_dict['class_%s'%i] = (self.labels_train == i)
#         idx = np.arange(len(self.labels_train))
#         idx = idx[self.class_dict['class_%s'%c]]
#         return np.random.permutation(idx)[:num]

#     def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
#         if self.class_dict2 is None:
#             self.class_dict2 = {}
#             for i in range(self.nclass):
#                 if transductive:
#                     idx = self.idx_train[self.labels_train == i]
#                 else:
#                     idx = np.arange(len(self.labels_train))[self.labels_train==i]
#                 self.class_dict2[i] = idx

#         if args.nlayers == 1:
#             sizes = [15]
#         if args.nlayers == 2:
#             sizes = [10, 5]
#             # sizes = [-1, -1]
#         if args.nlayers == 3:
#             sizes = [15, 10, 5]
#         if args.nlayers == 4:
#             sizes = [15, 10, 5, 5]
#         if args.nlayers == 5:
#             sizes = [15, 10, 5, 5, 5]


#         if self.samplers is None:
#             self.samplers = []
#             for i in range(self.nclass):
#                 node_idx = torch.LongTensor(self.class_dict2[i])
#                 self.samplers.append(NeighborSampler(adj,
#                                     node_idx=node_idx,
#                                     sizes=sizes, batch_size=num,
#                                     num_workers=12, return_e_id=False,
#                                     num_nodes=adj.size(0),
#                                     shuffle=False))
#         batch = np.random.permutation(self.class_dict2[c])[:num]
#         out = self.samplers[c].sample(batch)
#         return out

def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def calc_f1(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def evaluate(output, labels, args):
    data_graphsaint = ['yelp', 'ppi', 'ppi-large', 'flickr', 'reddit', 'amazon']
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        print("Test set results:", "F1-micro= {:.4f}".format(micro),
                "F1-macro= {:.4f}".format(macro))
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
    return


from torchvision import datasets, transforms
def get_mnist(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no        augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []
    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()
    from utils_graphsaint import GraphData
    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj-adj, feat, labels, idx, idx, idx)
    from deeprobust.graph.data import Dpr2Pyg
    return Dpr2Pyg(dpr_data)

def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1))/n - 0.5)

def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro

def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum()-thresh)
    # return F.relu(adj.sum()-thresh) / n**2

def feature_smoothing(adj, X):
    adj = (adj.t() + adj)/2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-8
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat

def row_normalize_tensor(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx


def loss_fn_kd(logits, logits_t):
    """This is the function of computing the soft target loss by using soft labels

    Args:
        logits (torch.Tensor): predictions of the student
        logits_t (torch.Tensor): logits generated by the teacher

    Returns:
        tuple: a tuple containing the soft target loss and the soft labels
    """

    loss_fn = nn.BCEWithLogitsLoss()

    # generate soft labels from logits
    labels_t = torch.where(logits_t > 0.0, 
                        torch.ones(logits_t.shape).to(logits_t.device), 
                        torch.zeros(logits_t.shape).to(logits_t.device)) 
    loss = loss_fn(logits, labels_t)

    return loss, labels_t