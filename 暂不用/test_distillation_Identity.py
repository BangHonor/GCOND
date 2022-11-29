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
import json
import time
import scipy
from models.gcn import GCN
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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
args = parser.parse_args()
args.dataset='reddit'
args.sgc=1
args.nlayers=2
args.lr_feat=0.1
args.lr_adj=0.1
args.r=0.002
args.reduction_rate=args.r
args.seed=1
args.gpu_id=0
args.epochs=1000
args.inner=1
args.outer=10
args.save=1

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

dataset_str='/home/xzb/GCond/data/'+args.dataset+'/'

loaded=np.load(dataset_str+'adj_full.npz')
#loaded is a csr matrix, contains 3 rows(indptr,indices,data), we can directly turn it into adj list

#get sparse format
try:
    matrix_format = loaded['format']
except KeyError as e:
    raise ValueError('The file {} does not contain a sparse matrix.'.format(file)) from e

matrix_format = matrix_format.item()
if not isinstance(matrix_format, str):
    # Play safe with Python 2 vs 3 backward compatibility;
    # files saved with SciPy < 1.0.0 may contain unicode or bytes.
    matrix_format = matrix_format.decode('ascii')
try:
    cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
except AttributeError as e:
    raise ValueError('Unknown matrix format "{}"'.format(matrix_format)) from e

#get adj list:
adj_full_list=[]
if matrix_format in ('csc', 'csr', 'bsr'):
    if matrix_format == 'csc':         
        print("稀疏矩阵格式为csc sparse")
    elif matrix_format == 'csr':         
        print("稀疏矩阵格式为csr sparse")
        data=loaded['data'],
        indice=loaded['indices']
        indptr=loaded['indptr']
        #print("data indice indptr的长度为：",len(data),len(indice),len(indptr))
        index=0
        for i in range(len(indptr)-1):#matrix len:232965
            temp=[]
            num_of_row=indptr[i+1]-indptr[i]
            for j in range(num_of_row):
                temp.append(indice[index])
                index+=1
            adj_full_list.append(np.array(temp))
    else:
        print("稀疏矩阵格式为bsr sparse")

parts=[1,3,5,10]
role = json.load(open(dataset_str+'role.json','r'))
idx_train = role['tr']
idx_test = role['te']
idx_val = role['va']

for part in parts:
    torch.cuda.empty_cache()
    print("将大图切分为",part,"个小图")
    start = time.perf_counter()
    print("切割开始！")
    n_cuts, membership = pymetis.part_graph(part, adjacency=adj_full_list)#receive adj ndarray list , return the group that every node belongs to
    end = time.perf_counter()
    print("切割时间为", round(end-start), '秒')
    nodes_part=[]
    for i in range(part):
        nodes_part.append(np.argwhere(np.array(membership) == i).ravel())
    idx_trains=[]
    idx_tests=[]
    idx_vals=[]
    for i in range(part):
        idx_trains.append([])
        idx_tests.append([])
        idx_vals.append([])

    #select index from the above three
    for i in idx_train:
        for j in range(part):
            if i in nodes_part[j]:
                idx_trains[j].append(i)
    for i in idx_test:
        for j in range(part):
            if i in nodes_part[j]:
                idx_tests[j].append(i)
    for i in idx_val:
        for j in range(part):
            if i in nodes_part[j]:
                idx_vals[j].append(i)
    # data = DataGraphSAINT(args.dataset)
    nclass=0
    feat = np.load(dataset_str+'feats.npy')
    feat_trains=[]
    adj_trains=[]
    label_trains=[]
    feat_vals=[]
    adj_vals=[]
    label_vals=[]
    t_models=[]
    dropout = 0.5 if args.dataset in ['reddit'] else 0

    for i in range(part):
        print("第",i,"个part：")
        print("training/val/test的size为",len(idx_trains[i]),len(idx_vals[i]),len(idx_tests[i]))
        data = DataGraphSAINT(args.dataset,idx_trains[i],idx_tests[i],idx_vals[i],label_rate=args.label_rate)
        nclass=data.nclass

        adj_train,feat_train=utils.to_tensor(data.adj_train,data.feat_train,labels=None,device='cuda')
        feat_trains.append(feat_train.detach())
        label_trains.append(torch.LongTensor(data.labels_train).to('cuda'))
        if utils.is_sparse_tensor(adj_train):
            adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
        else:
            adj_train = utils.normalize_adj_tensor(adj_train)
        adj_trains.append(adj_train.detach())

        t_models.append(GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        if os.path.exists('/home/xzb/GCond/saved_distillation/MLP_GCN_'+str(args.dataset)+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'_'+str(i)+'_'+str(part)+'_1000.pt'):
            t_models[i].load_state_dict(torch.load(f'/home/xzb/GCond/saved_distillation/MLP_GCN_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_1000.pt'))
        else:
            t_models[i].load_state_dict(torch.load(f'/home/xzb/GCond/saved_distillation/GCN_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_1000.pt'))
        print("读取GCN Teacher模型成功！")

    #now we get all teachers and student model, let's tam!
    stu_model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')
    stu_model.initialize()
    optimizer = optim.Adam(stu_model.parameters(), lr=0.01, weight_decay=5e-4)
        
    print("开始执行TAM算法！")
    eval_epoch=[20,40,60,80,120,160,200,300,400,500,600,700,800,900,1000]
    data = DataGraphSAINT(args.dataset,idx_train,idx_test,idx_val,label_rate=args.label_rate)
    feat_test, feat_val = data.feat_test,data.feat_val
    labels_test, labels_val =torch.LongTensor(data.labels_test).to('cuda'), torch.LongTensor(data.labels_val).to('cuda')
    adj_test, adj_val=data.adj_test, data.adj_val

    start = time.perf_counter()
    res = []
    for epoch in range(0,eval_epoch[-1]+1):
        loss_soft=torch.tensor(0.0).to('cuda')
        loss_hard=torch.tensor(0.0).to('cuda')
        loss_total=torch.tensor(0.0).to('cuda')
        T=1
        alpha=120
        for i in range(0,part):
            optimizer.zero_grad()
            t_T=t_models[i].forward_T(feat_trains[i],adj_trains[i],T)#所有数据需要在同一个设备上，所以先确保都是cuda上的tensor！
            stu_T=stu_model.forward_T(feat_trains[i],adj_trains[i],T)
            hard_labels=label_trains[i]
            stu_softmax=stu_model.forward(feat_trains[i],adj_trains[i])

            loss_fn=torch.nn.MSELoss(reduction='mean')
            loss=loss_fn(stu_T,t_T)
            loss_soft=loss_soft+loss

            loss_fn=torch.nn.NLLLoss()
            loss=loss_fn(stu_softmax,hard_labels)
            loss_hard=loss_hard+loss

            loss_total=loss_soft+alpha*loss_hard

        loss_total.backward()
        optimizer.step()
        if epoch in eval_epoch:
            print("epoch:",epoch,"validation set!")
            output=stu_model.predict(feat_val,adj_val)
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
    print("TAM算法执行结束！")
    torch.save(stu_model.state_dict(), f'saved_distillation/GCN_Student_{args.dataset}_{args.reduction_rate}_{args.seed}_{part}.pt') 

    output=stu_model.predict(feat_test,adj_test)
    loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    print("Test set results of amalgamated student model:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))

