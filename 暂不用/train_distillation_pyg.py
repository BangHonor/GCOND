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
from models_pyg.gcn import GCN,train,test
# from models_pyg.gat import GAT
# from models_pyg.sgc import SGC
# from models_pyg.mygraphsage import GraphSage
# from models_pyg.myappnp1 import APPNP1
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
import torch_geometric.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
args = parser.parse_args() 
args.dataset='ogbn-arxiv'
args.sgc=1
args.nlayers=3
args.model='GCN'
args.hidden=256
args.reduction_rate=0.001
args.seed=1
args.gpu_id=1
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
pyg_dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                    transform=T.ToSparseTensor())
pyg_data=pyg_dataset[0]
pyg_data=pyg_data.to('cuda')
adj_t=pyg_data.adj_t
adj_t = adj_t.to_symmetric()
split_idx = pyg_dataset.get_idx_split()

dpr_data = get_dataset(args.dataset, args.normalize_features)#dpr继承了pyg的属性！！！！！！！！！！！！！
idx_train, idx_val, idx_test = dpr_data.idx_train, dpr_data.idx_val, dpr_data.idx_test
data = Transd2Ind_partition(dpr_data.features,dpr_data.adj,dpr_data.labels,idx_train,idx_test,idx_val,keep_ratio=args.keep_ratio)
feat_test, feat_val = torch.FloatTensor(data.feat_test).to('cuda'),torch.FloatTensor(data.feat_val).to('cuda')
labels_test, labels_val =torch.LongTensor(data.labels_test).to('cuda'), torch.LongTensor(data.labels_val).to('cuda')

loaded=dpr_data.adj
role_map=[0]*(len(idx_train)+len(idx_test)+len(idx_val))
for i in idx_test:
    role_map[int(i)]=1
for i in idx_val:
    role_map[int(i)]=2

#csr稀疏矩阵
print("稀疏矩阵格式为csr sparse")
indice=loaded.indices#通过属性而不是[]获取
indptr=loaded.indptr
adj_full_list=[]
#print("data indice indptr的长度为：",len(data),len(indice),len(indptr))
index=0
for i in range(len(indptr)-1):#matrix len:232965
    temp=[]
    num_of_row=indptr[i+1]-indptr[i]
    for j in range(num_of_row):
        temp.append(indice[index])
        index+=1
    adj_full_list.append(np.array(temp))

parts=[1,3,5]
for part in parts:
    #切割，然后把feature、adj、label、index等传进去
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
    for i in range(part):
        for j in nodes_part[i]:
            j=int(j)
            if(role_map[j]==0):
                idx_trains[i].append(j)
            elif(role_map[j]==1):
                idx_tests[i].append(j)
            else:
                idx_vals[i].append(j)
    
    nclass=0
    feat = dpr_data.features
    feat_trains=[]
    adj_trains=[]
    label_trains=[]
    t_models=[]
    feat_syns=[]
    adj_syns=[]
    label_syns=[]
    
    for i in range(part):
        print("第",i,"个part：")
        print("training/val/test的size为",len(idx_trains[i]),len(idx_vals[i]),len(idx_tests[i]))
        data = Transd2Ind_partition(dpr_data.features,dpr_data.adj,dpr_data.labels,idx_trains[i],idx_tests[i],idx_vals[i],keep_ratio=args.keep_ratio)
        nclass=data.nclass

        adj_train,feat_train=utils.to_tensor(data.adj_train,data.feat_train,labels=None,device='cuda')
        feat_trains.append(feat_train.detach())
        label_trains.append(torch.LongTensor(data.labels_train).to('cuda'))
        if utils.is_sparse_tensor(adj_train):
            adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
        else:
            adj_train = utils.normalize_adj_tensor(adj_train)
        adj_trains.append(adj_train.detach())

        start = time.perf_counter()
        agent = GCond(data, args, device='cuda')
        label_syns.append(agent.labels_syn)
        # agent.train(i,part)
         # agent.val_with_syn(i,part,args.model)
        end = time.perf_counter()
        print("第i个小图凝聚用时:",round(end-start), '秒')

        if args.model=='GCN':
            t_models.append(GCN(feat.shape[1], args.hidden, nclass, args.nlayers, args.dropout).to('cuda'))
        elif args.model=='SGC':
            t_models.append(SGC(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        elif args.model=='GraphSage':
            t_models.append(GraphSage(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        elif args.model=='APPNP1':
            t_models.append(APPNP1(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        
        if os.path.exists('/home/xzb/GCond/saved_distillation/'+str(args.dataset)+'_'+str(args.dataset)+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'_'+str(i)+'_'+str(part)+'_1000.pt'):
            t_models[i].load_state_dict(torch.load(f'/home/xzb/GCond/saved_distillation/{args.model}_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_1000.pt'))
        print("读取Teacher模型成功！")
        feat_syns.append(torch.load(f'/home/xzb/GCond/saved_distillation/feat_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_1000.pt', map_location=torch.device('cuda')))
        #2*edge形式
        adj_syns.append(torch.ones((2,feat_syns[i].shape[0]),dtype=torch.long).to('cuda'))

    #now we get all teachers and student model, let's tam!
    if args.model=='GCN':
        stu_model = GCN(feat.shape[1], args.hidden, pyg_dataset.num_classes, args.nlayers, args.dropout).to('cuda')
    elif args.model=='SGC':
        stu_model = SGC(feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')
    elif args.model=='GraphSage':
        stu_model = GraphSage(feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')
    else:
        stu_model = APPNP1(feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')
    stu_model.reset_parameters()
    optimizer = optim.Adam(stu_model.parameters(), lr=0.01, weight_decay=5e-4)
        
    print("开始执行TAM算法！")
    start = time.perf_counter()
    res = []
    evaluator = Evaluator(name='ogbn-arxiv')
    for epoch in range(0,args.epochs):
        stu_model.train()
        optimizer.zero_grad()
        loss_soft=torch.tensor(0.0).to('cuda')
        loss_hard=torch.tensor(0.0).to('cuda')
        loss_total=torch.tensor(0.0).to('cuda')
        for i in range(0,part):
            # t_T=t_models[i].forward(pyg_data.x[idx_trains[i]],adj_t[idx_trains[i],idx_trains[i]])#1G
            # stu_T=stu_model.forward(pyg_data.x[idx_trains[i]],adj_t[idx_trains[i],idx_trains[i]])#0.5G
            # hard_labels=label_trains[i]
            # stu_softmax=stu_model.forward(pyg_data.x[idx_trains[i]],adj_t[idx_trains[i],idx_trains[i]])#0.4G
            #test syn
            #用稀疏矩阵或者dgl样式edge_index表示
            # t_T=t_models[i].forward_T(feat_syns[i],adj_syns[i],T)#1G
            # stu_T=stu_model.forward_T(feat_syns[i],adj_syns[i],T)#0.5G
            # hard_labels=label_syns[i]
            # stu_softmax=stu_model.forward(feat_syns[i],adj_syns[i])#0.4G
        
            # loss_fn=torch.nn.MSELoss(reduction='mean')
            # loss_soft=loss_fn(stu_T,t_T)

            # loss_hard= F.nll_loss(stu_softmax, pyg_data.y.squeeze(1)[idx_trains[i]])
            # acc_train = utils.accuracy(stu_softmax, pyg_data.y.squeeze(1)[idx_trains[i]])
            loss_total = train(stu_model, pyg_data, idx_trains[i], optimizer)
        # loss_hard.backward()
        # optimizer.step()

        if epoch%5==0:
            train_acc, valid_acc, test_acc=test(stu_model, pyg_data, split_idx, evaluator)
            # stu_model.eval()
            # out = stu_model(pyg_data.x, pyg_data.adj_t)
            # y_pred = out.argmax(dim=-1, keepdim=True)
            # train_acc = evaluator.eval({
            #     'y_true': pyg_data.y[idx_train],
            #     'y_pred': y_pred[idx_train],
            # })['acc']
            # valid_acc = evaluator.eval({
            #     'y_true': pyg_data.y[idx_val],
            #     'y_pred': y_pred[idx_val],
            # })['acc']
            # test_acc = evaluator.eval({
            #     'y_true': pyg_data.y[idx_test],
            #     'y_pred': y_pred[idx_test],
            # })['acc']
            print(  f'Epoch: {epoch:02d}, '
                    f'Loss: {loss_total:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')

    end = time.perf_counter()
    print("知识蒸馏时长:",round(end-start), '秒')
    print("TAM算法执行结束！")
    torch.save(stu_model.state_dict(), f'/home/xzb/GCond/saved_distillation/{args.model}_Student_{args.dataset}_{args.reduction_rate}_{args.seed}_{part}.pt') 

