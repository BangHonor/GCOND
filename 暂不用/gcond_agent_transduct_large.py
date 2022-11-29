import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborSampler
import gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        d = data.features.shape[1]#feat_train是大图X 一个节点特征向量的维度
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)#得到小图的所有label
        self.nnodes_syn = len(self.labels_syn)
        n = self.nnodes_syn

        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))#feat_syn是小图X'，随机生成可训练参数
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)#X'得到A'的算法,参数是φ

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)#每个class进行数数量统计 字典
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])#对次数进行排序,每一个元素为{class,n}
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):#to make num of labels_syn=counter*redcution_rate 
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def train(self, train_loader):#开始训练
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn#小图的X' A' Y'

        #获得了小图的X' 正则化后的A' Y',开始梯度下降
        best_acc=0
        outer_loop, inner_loop = get_loops(args)#获得里外循环的超参数
        print("开始图凝聚！")
        for it in range(args.epochs+1):
            loss_avg = 0
            if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
                model = SGC1(nfeat=feat_syn.shape[1], nhid=self.args.hidden,
                            dropout=0.0, with_bn=False,
                            weight_decay=0e-4, nlayers=2,
                            nclass=data.nclass,
                            device=self.device).to(self.device)
            else:
                if args.sgc == 1:
                    model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                                nclass=data.nclass, dropout=args.dropout,
                                nlayers=args.nlayers, with_bn=False,
                                device=self.device).to(self.device)
                else:
                    model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,#GCN的输入 可以看到和节点的个数没有关系 layers是GCN的数量 hidden是每一个GCN层的输入输出维度：nfeat->nhidden->nhidden->nclass
                                nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                                device=self.device).to(self.device)#把模型加载到CUDA上

            model.initialize()#对参数进行初始化
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)#model只包括GCN里面的参数
            model.train()#BN+DropOut
            
            for batch_size, n_id, adjs in train_loader:
                labels_train =  torch.LongTensor(data.labels[n_id[:batch_size]]).to(self.device)
                adjs=[adj.to(self.device) for adj in adjs]

                for ol in range(outer_loop):
                    adj_syn = pge(self.feat_syn)#PGE算法得到小图的邻接矩阵，小图不能太大，否则PGE的GPU内存会爆
                    adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)

                    loss = torch.tensor(0.0).to(self.device)
                    output_syn = model.forward(feat_syn, adj_syn_norm)#small, full batch forward is ok
                    output = model.forward_sampler_large(torch.FloatTensor(data.features[n_id]).to(self.device), adjs)#大图的output 里面有BN+DROPOUT
                    loss_real = F.nll_loss(output, labels_train)#大图的CrossEntropy
                    gw_real = torch.autograd.grad(loss_real, model_parameters)#作用：计算并返回outputs对w的梯度
                    gw_real = list((_.detach().clone() for _ in gw_real))#通过 tensor.detach().clone() 操作生成一个和原本 tensor 值相同的新 tensor

                    #小图
                    loss_syn = F.nll_loss(output_syn,labels_syn)#小图的CrossEntropy，只计算两个？
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    loss = match_loss(gw_syn, gw_real, args, device=self.device)#梯度匹配算法

                    #计算所有分类的梯度差和
                    loss_avg += loss.item()#tensor的值
                    # TODO: regularize
                    if args.alpha > 0:
                        loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                    else:
                        loss_reg = torch.tensor(0)

                    loss = loss + loss_reg#所有的类总的loss，根据loss做梯度下降

                    # update sythetic graph
                    self.optimizer_feat.zero_grad()#X'梯度设置为0
                    self.optimizer_pge.zero_grad()#φ梯度设置为0
                    loss.backward()#loss是两幅图放入GCN中的output loss!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if it % 50 < 10:#loss对X'和φ的梯度 为了使loss最小 我们最优化X'和φ'
                        self.optimizer_pge.step()
                    else:
                        self.optimizer_feat.step()

                    if args.debug and ol % 5 ==0:
                        print('Gradient matching loss:', loss.item())

                    if ol == outer_loop - 1:
                        break
                    
                    #不是直接对θ梯度下降，而是得到新的小图后，再代进去求新的loss然后梯度下降
                    feat_syn_inner = feat_syn.detach()
                    adj_syn_inner = pge.inference(feat_syn_inner)#φ x'-(pge)->a'
                    adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                    feat_syn_inner_norm = feat_syn_inner
                    for j in range(inner_loop):#why mutiple? one step for xita is no enough to change the loss
                        optimizer_model.zero_grad()
                        output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                        loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                        loss_syn_inner.backward()
                        optimizer_model.step() # update gnn param θ！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            #one epoch finished, the avg loss for every class in this epoch
            loss_avg /= outer_loop
            eval_epochs = [0,10,20,30,40,50,60,70,80,90,100]

            if it in eval_epochs:#condensed for certain epoches, use val data to
                print('Epoch {}, match loss_avg: {}'.format(it, loss_avg))
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M'] else 3
                for i in range(runs):
                    if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
                        res.append(self.test_with_val(args, best_acc))
                        best_acc=max(best_acc,res[0][1])
                #多个runs的情况

    def test_with_val(self, args, best_acc, verbose=True):
        #features是numpy adj是SparseTensor
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model.fit_with_val(args, feat_syn, adj_syn, labels_syn, data,#fit training gcn model, not condensed graph
                     train_iters=600, normalize=True, large=True, verbose=False, noval=True)

        model.eval()
        # Full graph
        # output = model.predict(data.feat_full, data.adj_full)#data.feat_full:2708*1433

        feat_train, adj_train=data.feat_train, data.adj_train
        labels_train = torch.LongTensor(data.labels[data.idx_train]).to(self.device)
        res.append(model.train_test_acc(args, feat_train, adj_train, labels_train, flag='train'))
        del feat_train, adj_train, labels_train
        gc.collect()

        feat_test, adj_test=data.feat_test, data.adj_test
        labels_test = torch.LongTensor(data.labels[data.idx_test]).to(self.device)
        res.append(model.train_test_acc(args, feat_test, adj_test, labels_test, flag='test'))
        del feat_test, adj_test, labels_test
        gc.collect()

        if self.args.save and res[1]>best_acc:
            torch.save(adj_syn, f'/home/xzb/GCond/saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')#save X' A'
            torch.save(feat_syn, f'/home/xzb/GCond/saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(model.state_dict(), f'/home/xzb/GCond/saved_model/{args.model}_{args.dataset}_{args.reduction_rate}_{args.seed}.pt') 
        return res

    def get_sub_adj_feat(self, features):#获取相邻接点的特征
        data = self.data#data包含了输入的X' train、test的index、feature等
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())#self.labels_syn类似于{分类1，分类1，分类1，分类2，分类2，分类3}，统计了小图每个分类的数量

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])#检索小图对应class的节点index
            tmp = list(tmp)
            idx_selected = idx_selected + tmp#每一行都是各自分类选择的节点index
        idx_selected = np.array(idx_selected).reshape(-1)#化为一行
        features = features[self.data.idx_train][idx_selected]#在train中选部分，因为idx_train是index中最前的一批，所以不会选择到test的index

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())#计算小图中feature的两两距离
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0#对角线初始化为0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
            sims[i, indices_argsort[: -k]] = 0#除了k个相邻的（余弦最大的） 其他设置为0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    # if args.one_step:
    #     if args.datasetin ['ogbn-arxiv','ogbn-products']:
    #         return 5, 0
    #     return 1, 0
    if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15 # sgc
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10

