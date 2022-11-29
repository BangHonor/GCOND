from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import deeprobust.graph.utils as utils
import torch.nn.functional as F
from GCond.暂不用.gcond_agent_transduct_large import GCond
from utils_graphsaint_partition import DataGraphSAINT
import pymetis
import scipy.sparse as sp
import json
import time
import scipy
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
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbn-papers100M')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=100)#one epoch means update condensed graph once and gcn model for multiple times
parser.add_argument('--nlayers', type=int, default=2)#layers of gcn
parser.add_argument('--hidden', type=int, default=256)#columns of w matrix
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)#L2
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)#
parser.add_argument('--reduction_rate', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--model', type=str, default='GCN')
args = parser.parse_args()
print(args)
torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data = Transd2Ind_Large(args.dataset, args.keep_ratio)

start = time.perf_counter()
agent = GCond(data, args, device='cuda')
agent.train(data.train_loader)
end = time.perf_counter()
print('图凝聚用时:',round(end-start), '秒')
