from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from GCond.暂不用.gcond_agent_induct_partition_Identity import GCond
from utils_graphsaint_partition import DataGraphSAINT
import pymetis
import scipy.sparse as sp
import time
import scipy
from models.gcn import GCN
from models.mygraphsage import GraphSage
from models.myappnp1 import APPNP1
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import match_loss, regularization, row_normalize_tensor,loss_fn_kd
import deeprobust.graph.utils as utils
import os
import csv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
from torch_geometric.data import NeighborSampler

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.001)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
args = parser.parse_args() 
args.dataset='ogbn-arxiv'
args.model='GCN'
args.nlayers=2
args.hidden=256
args.lr=0.01

torch.cuda.set_device(args.gpu_id)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dpr_data = get_dataset(args.dataset, args.normalize_features)
idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
role_map=[0]*(len(idx_train)+len(idx_test)+len(idx_val))

nclass=0
feat = dpr_data.features
features=dpr_data.features
labels=dpr_data.labels
adj=dpr_data.adj
features, adj, labels = utils.to_tensor(features, adj, labels, device='cuda')
if utils.is_sparse_tensor(adj):
    adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
else:
    adj_norm = utils.normalize_adj_tensor(adj)

adj = adj_norm
adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
        value=adj._values(), sparse_sizes=adj.size()).t()

data = Transd2Ind_partition(dpr_data.features,dpr_data.adj,dpr_data.labels,idx_train,idx_test,idx_val,keep_ratio=args.keep_ratio)
nclass=data.nclass

adj_train,feat_train=utils.to_tensor(data.adj_train,data.feat_train,labels=None,device='cuda')
if utils.is_sparse_tensor(adj_train):
    adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
else:
    adj_train = utils.normalize_adj_tensor(adj_train)

dropout = 0.5
if args.model=='GCN':
    model=GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=args.nlayers,nclass=nclass, device='cuda').to('cuda')
elif args.model=='SGC':
    model=SGC(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=args.nlayers,nclass=nclass, device='cuda').to('cuda')
elif args.model=='GraphSage':
    model=GraphSage(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=args.nlayers,nclass=nclass, device='cuda').to('cuda')
elif args.model=='APPNP1':
    model=APPNP1(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=args.nlayers,nclass=nclass, device='cuda').to('cuda')
else:
    model=GAT(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=args.nlayers,nclass=nclass, device='cuda').to('cuda')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
model.initialize()#每一次循环都不一样

print("开始训练！")
eval_epoch=[20,40,60,80,120,140,160,180,200,300,400,500,600,700,800,900,1000]
feat_test, feat_val = data.feat_test,data.feat_val
labels_test, labels_val =torch.LongTensor(data.labels_test).to('cuda'), torch.LongTensor(data.labels_val).to('cuda')
adj_test, adj_val=data.adj_test, data.adj_val

start = time.perf_counter()
res = []
for epoch in range(0,eval_epoch[-1]+1):
    model.train()#开启BN+DROPOUT 如果model定义的时候没有BN+DROPOUT将无效
    loss_hard=torch.tensor(0.0).to('cuda')
    optimizer.zero_grad()
    hard_labels=torch.LongTensor(data.labels_train).to('cuda')
    stu_softmax=model.forward(feat_train,adj_train)#0.4G

    loss_fn=torch.nn.NLLLoss()
    loss_hard=loss_fn(stu_softmax,hard_labels)
    loss_hard.backward()
    optimizer.step()

    # loss = torch.tensor(0.0).to('cuda')
    # optimizer.zero_grad()
    # for c in range(data.nclass):
    #     batch_size, n_id, adjs = data.retrieve_class_sampler(#neighborsampler
    #             c, adj, transductive=False, args=args)
    #     #print(c,"分类计算完毕！")
    #     if args.nlayers == 1:
    #         adjs = [adjs]
    #     adjs = [adj.to('cuda') for adj in adjs]
    #     output = model.forward_sampler(features[n_id], adjs)#计算的只是这个类作为input的output
    #     loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
    #     loss=loss+loss_real
    # loss.backward()
    # optimizer.step()
    if epoch in eval_epoch:
        model.eval()
        print("epoch:",epoch,"validation set!")
        output=model.predict(feat_val,adj_val)
        loss_val = F.nll_loss(output, labels_val)
        acc_val = utils.accuracy(output, labels_val)
        res.append(acc_val.item())
        print("Val set results of amalgamated student model:",
                "loss= {:.4f}".format(loss_val.item()),
                "accuracy= {:.4f}".format(acc_val.item()))
        if len(res)>1 and res[-1]<res[-2]:
            break
        
end = time.perf_counter()
print("知识蒸馏时长:",round(end-start), '秒')
print("全图训练结束！")

model.eval()
output=model.predict(feat_test,adj_test)
loss_test = F.nll_loss(output, labels_test)
acc_test = utils.accuracy(output, labels_test)
print("Test set results of amalgamated student model:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))
torch.save(model.state_dict(), f'/home/xzb/GCond/saved_distillation/{args.model}_Whole_{args.dataset}.pt') 