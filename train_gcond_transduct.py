from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcond_agent_transduct import GCond
from utils_graphsaint_partition import DataGraphSAINT
import pymetis
import scipy.sparse as sp
import json
import time
import scipy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import match_loss, regularization, row_normalize_tensor,loss_fn_kd
import deeprobust.graph.utils as utils
import os
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=3, help='gpu id')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=1000)#one epoch means update condensed graph once and gcn model for multiple times
parser.add_argument('--nlayers', type=int, default=0)
parser.add_argument('--hidden', type=int, default=256)#columns of w matrix
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)#L2
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)#
parser.add_argument('--reduction_rate', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=10)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--model', type=str, default='GCN')

#deepgcn
# parser.add_argument('--deep_layers', type=int, default=28,
#                             help='the number of layers of the networks')
# parser.add_argument('--deep_hidden', type=int, default=128)
# parser.add_argument('--mlp_layers', type=int, default=1,
#                             help='the number of layers of mlp in conv')
# parser.add_argument('--block', default='res+', type=str,
#                             help='graph backbone block type {res+, res, dense, plain}')
# parser.add_argument('--conv', type=str, default='gen',
#                             help='the type of GCNs')
# parser.add_argument('--gcn_aggr', type=str, default='softmax_sg',
#                             help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, softmax_sum, power, power_sum]')
# parser.add_argument('--t', type=float, default=0.1,
#                             help='the temperature of SoftMax')
# parser.add_argument('--p', type=float, default=1.0,
#                     help='the power of PowerMean')
# parser.add_argument('--y', type=float, default=0.0,
#                     help='the power of degrees')
# parser.add_argument('--learn_t', action='store_true')
# parser.add_argument('--learn_p', action='store_true')
# parser.add_argument('--learn_y', action='store_true')
# parser.add_argument('--norm', type=str, default='batch',
#                             help='the type of normalization layer')
# # message norm
# parser.add_argument('--msg_norm', type=bool, default=True)
# parser.add_argument('--learn_msg_scale', type=bool, default=True)
# # save model
# parser.add_argument('--model_save_path', type=str, default='model_ckpt',
#                     help='the directory used to save models')
# # load pre-trained model
# parser.add_argument('--model_load_path', type=str, default='ogbn_products_pretrained_model.pth',
#                     help='the path of pre-trained model')

args = parser.parse_args()
torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

data_full = get_dataset(args.dataset, args.normalize_features)#get a Pyg2Dpr class, contains all index, adj, labels, features
data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)#transductive to inductive 同时实现了neighborsampler
agent = GCond(data, args, device='cuda')

start = time.perf_counter()
agent.train()
end = time.perf_counter()
print('图凝聚用时:',round(end-start), '秒')
