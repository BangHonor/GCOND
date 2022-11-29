from threading import local
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
from ka_loss import optimizing, loss_fn_kd, gen_mi_attrib_loss
import torch.autograd as autograd
import gc

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
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
#topo parse
parser.add_argument("--residual", action="store_true", default=True,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=0,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0,
                    help="attention dropout")
parser.add_argument('--alpha', type=float, default=0.2,
                    help="the negative slop of leaky relu")
parser.add_argument('--batch-size', type=int, default=2,
                    help="batch size used for training, validation and test")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument("--s-epochs", type=int, default=1500,
                    help="number of training epochs")
parser.add_argument('--attrib-weight', type=float, default=0.10,
                    help="weight coeff of the topological semantics alignment loss")
args = parser.parse_args()
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = parser.parse_args()
args.dataset='reddit'
args.sgc=1
args.nlayers=2
args.lr_feat=0.1
args.lr_adj=0.1
args.reduction_rate=0.005
args.seed=1
args.gpu_id=0
args.epochs=1000
args.inner=1
args.outer=10
args.save=0
args.attrib_weight=0.01

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
    idx_fulls=[]
    part_dict_train=np.arange(len(idx_train)+len(idx_val)+len(idx_test))
    part_dict_full=np.arange(len(idx_train)+len(idx_val)+len(idx_test))
    for i in range(len(part_dict_train)):
        part_dict_train[i]=-1

    for i in range(part):
        idx_trains.append([])
        idx_tests.append([])
        idx_vals.append([])
        idx_fulls.append([])
        for j in nodes_part[i]:
            part_dict_full[j]=i

    #select index from the above three
    for i in idx_train:
        idx_trains[part_dict_full[i]].append(i)
        idx_fulls[part_dict_full[i]].append(i)
        part_dict_train[i]=part_dict_full[i]

    for i in idx_test:
        idx_vals[part_dict_full[i]].append(i)
        idx_fulls[part_dict_full[i]].append(i)
    for i in idx_val:
        idx_tests[part_dict_full[i]].append(i)
        idx_fulls[part_dict_full[i]].append(i)

    #construct dgl, only training data
    # u=[]
    # v=[]
    # a=[]
    # for i in range(part):
    #     u.append([])
    #     v.append([])
    # i=0
    # for first_node in range(len(adj_full_list)):
    #     i=i+1
    #     if(i>1000):
    #         break
    #     for second_node in adj_full_list[first_node]:
    #         if part_dict_train[first_node]!=-1 and part_dict_train[first_node]==part_dict_train[second_node]:
    #             u[part_dict_train[first_node]].append(torch.tensor(first_node).to('cuda'))
    #             v[part_dict_train[first_node]].append(torch.tensor(second_node).to('cuda'))
    #idx_train_map=np.arange(len(idx_train)+len(idx_test)+len(idx_val))
    # for i in range(part):
    #     a.append(torch.ones(1,len(u[i]),requires_grad=True).to('cuda'))
    #     for j in range(len(u[i])):
    #         u[i][j]=idx_train_map[u[i][j]]
    #         v[i][j]=idx_train_map[v[i][j]]
    #     for j in range(len(idx_trains[i])):
    #         idx_train_map[idx_trains[i][j]]=j


    nclass=0
    feat = np.load(dataset_str+'feats.npy')
    feat_trains=[]
    adj_trains=[]
    label_trains=[]
    t_models=[]
    dropout = 0.5 if args.dataset in ['reddit'] else 0
    best_it=[]
    if part==3:
        best_it.extend([1000,400,200])
    elif part==5:
        best_it.extend([600,400,400,800,800])
    else:
        best_it.extend([400])

    for i in range(part):
        torch.cuda.empty_cache()
        print("第",i,"个part：")
        print("training/val/test的size为",len(idx_trains[i]),len(idx_vals[i]),len(idx_tests[i]))
        data = DataGraphSAINT(args.dataset,idx_trains[i],idx_tests[i],idx_vals[i],label_rate=args.label_rate)
        nclass=data.nclass

        adj_train,feat_train=utils.to_tensor(data.adj_train,data.feat_train,labels=None,device='cuda')
        #map u,v into corresponded index with feature and adj

        feat_trains.append(feat_train.detach())
        label_trains.append(torch.LongTensor(data.labels_train).to('cuda'))
        if utils.is_sparse_tensor(adj_train):
            adj_train = utils.normalize_adj_tensor(adj_train, sparse=True)
        else:
            adj_train = utils.normalize_adj_tensor(adj_train)
        adj_trains.append(adj_train)

        t_models.append(GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda'))
        t_models[i].load_state_dict(torch.load(f'/home/xzb/GCond/saved_ours/GCN_{args.dataset}_{args.reduction_rate}_{args.seed}_{i}_{part}_{best_it[i]}.pt'))
        print("读取GCN Teacher模型成功！")

    #now we get all teachers and student model, let's tam!
    stu_model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=dropout,weight_decay=5e-4, nlayers=2,nclass=nclass, device='cuda').to('cuda')
    stu_model.initialize()
    last=-1
    # stu_model.load_state_dict(torch.load(f'/home/xzb/GCond/saved_ours/stu_{last}_{part}_{args.attrib_weight}.pt'))
    optimizer = optim.Adam(stu_model.parameters(), lr=0.01, weight_decay=5e-4)
        
    print("开始执行TAM算法！")
    eval_epoch=np.arange(0,1000,10)
    data = DataGraphSAINT(args.dataset,idx_train,idx_test,idx_val,label_rate=args.label_rate)
    feat_test = data.feat_test
    labels_test =torch.LongTensor(data.labels_test).to('cuda')
    adj_test=data.adj_test

    for epoch in range(last+1,2001):
        torch.cuda.empty_cache()
        print("epoch",epoch)
        loss_soft=torch.tensor(0.0).to('cuda')
        loss_topo=torch.tensor(0.0).to('cuda')
        loss_total=torch.tensor(0.0).to('cuda')
        
        for i in range(part):
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            #soft loss
            #adj_trains[i].requires_grad=True
            feat_trains[i].requires_grad=True
            stu_logits=stu_model.forward_logits(feat_trains[i],adj_trains[i])#adj太大的话跑不了，所以要subgraph，dgl会更方便
            t_logits=t_models[i].forward_logits(feat_trains[i],adj_trains[i])
            # loss,label=loss_fn_kd(stu_logits,t_logits)
            # loss_soft=loss_soft+loss
            loss_fn=torch.nn.MSELoss(reduction='mean')
            loss_soft=loss_fn(stu_logits,t_logits)

            #topo loss
            # stu_logits=stu_model.forward_sparse(feat_trains[i],a[i],u[i],v[i])#adj太大的话跑不了，所以要subgraph，dgl会更方便
            # t_logits=t_models[i].forward_sparse(feat_trains[i],a[i],u[i],v[i])
            labels_t = torch.where(t_logits > 0.0, torch.ones(t_logits.shape).to('cuda'), torch.zeros(t_logits.shape).to('cuda')).type(torch.bool)
            labels_s = torch.where(stu_logits > 0.0, torch.ones(stu_logits.shape).to('cuda'), torch.zeros(stu_logits.shape).to('cuda')).type(torch.bool)
            output_grad_t = torch.zeros(t_logits.shape,requires_grad=True).to('cuda')
            output_grad_s = torch.zeros(stu_logits.shape,requires_grad=True).to('cuda')
            output_grad_t[labels_t] = 1
            output_grad_s[labels_s] = 1
            attrib_map_t = autograd.grad(outputs=t_logits, inputs=feat_trains[i], grad_outputs=output_grad_t, only_inputs=True, retain_graph=True)[0]
            attrib_map_s = autograd.grad(outputs=stu_logits, inputs=feat_trains[i], grad_outputs=output_grad_s, only_inputs=True, retain_graph=True)[0]
            attrib_map_t.requires_grad=True
            attrib_map_s.requires_grad=True
            loss_topo=loss_fn(attrib_map_s,attrib_map_t)
            loss_total = loss_total+loss_soft + args.attrib_weight * loss_topo#define weight by yourself
        loss_total.backward()
        optimizer.step()
        del loss_soft,loss_topo,loss_total
        if epoch%10==0:
            torch.save(stu_model.state_dict(), f'/home/xzb/GCond/saved_ours/stu_{epoch}_{part}_{args.attrib_weight}.pt')#store syn grpah
        if epoch in eval_epoch:
            print("epoch:",epoch,"test model")
            output=stu_model.predict(feat_test,adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = utils.accuracy(output, labels_test)
            res = []
            res.append(acc_test.item())
            print("Test set results of amalgamated student model:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))
            del output,loss_test,acc_test,res
        gc.collect()
    print("TAM算法执行结束！")
    #torch.save(stu_model.state_dict(), f'saved_ours/GCN_Student.pt') 

    