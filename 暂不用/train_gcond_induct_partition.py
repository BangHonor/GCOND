from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from GCond.暂不用.gcond_agent_induct_partition import GCond
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
parser.add_argument('--reduction_rate', type=float, default=0.01)
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

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

dataset_str='data/'+args.dataset+'/'

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

parts=[5,10]
role = json.load(open(dataset_str+'role.json','r'))
idx_train = role['tr']
idx_test = role['te']
idx_val = role['va']

for part in parts:
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
    best_it=[]#对应part的数量
    nclass=0
    feat = np.load(dataset_str+'feats.npy')
    feat_trains=[]
    adj_trains=[]
    t_models=[]
    dropout = 0.5 if args.dataset in ['reddit'] else 0
    stu_model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')

    for i in range(part):
        print("第",i,"个part：")
        print("training/val/test的size为",len(idx_trains[i]),len(idx_vals[i]),len(idx_tests[i]))
        data = DataGraphSAINT(args.dataset,idx_trains[i],idx_tests[i],idx_vals[i],label_rate=args.label_rate)
        nclass=data.nclass
        feat_trains.append(data.feat_train)
        adj_trains.append(data.adj_train)
        #print(data.feat_train.shape,data.adj_train.shape)
        adj_trains[i],feat_trains[i]=utils.to_tensor(adj_trains[i],feat_trains[i],labels=None,device='cuda')
        if utils.is_sparse_tensor(adj_trains[i]):
            adj_trains[i] = utils.normalize_adj_tensor(adj_trains[i], sparse=True)
        else:
            adj_trains[i] = utils.normalize_adj_tensor(adj_trains[i])
        
        agent = GCond(data, args, device='cuda')
        t_models.append(GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        best_it.append(agent.train(i,part))
        t_models[i].load_state_dict(torch.load(f'saved_ours/GCN_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_{best_it[i]}.pt'))
        if i==0:
            stu_model=t_models[i]
        print("读取GCN模型成功！")

    #now we get all teachers and student model, let's tam!
    loss_soft=torch.tensor(0.0).to('cuda')
    loss_topo=torch.tensor(0.0).to('cuda')
    optimizer = optim.Adam(stu_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("开始执行TAM算法！")
    for epoch in range(0,10):
        for i in range(1,part):
            feat_trains[i]=feat_trains[i].detach()
            adj_trains[i]=adj_trains[i].detach()

            optimizer.zero_grad()
            t_logits=t_models[i].predict(feat_trains[i],adj_trains[i],logits=True).detach()#所有数据需要在同一个设备上，所以先确保都是cuda上的tensor！
            stu_logits=stu_model.predict(feat_trains[i],adj_trains[i],logits=True)
            loss_soft,label=loss_fn_kd(stu_logits,t_logits)
            loss_soft.requires_grad(required_grad=True)
            loss_soft.backward()
            optimizer.step()
    print("TAM算法执行结束！")
    #torch.save(stu_model.state_dict(), f'saved_ours/GCN_Student.pt') 

    data = DataGraphSAINT(args.dataset,idx_train,idx_test,idx_val,label_rate=args.label_rate)
    feat_test = data.feat_test
    labels_test =torch.LongTensor(data.labels_test).to('cuda')
    adj_test=data.adj_test
    output=stu_model.predict(feat_test,adj_test)
    loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    res = []
    res.append(acc_test.item())
    print("Test set results of amalgamated student model:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))