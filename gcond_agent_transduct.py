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
from models.deepgcn import DeeperGCN
import scipy.sparse as sp
from torch_sparse import SparseTensor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        d = data.feat_train.shape[1]#feat_train是大图X 一个节点特征向量的维度
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)#得到小图的所有label
        self.nnodes_syn = len(self.labels_syn)
        n = self.nnodes_syn

        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))#feat_syn是小图X'，随机生成可训练参数
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)#X'得到A'的算法,参数是φ
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)

        self.adj_syn=torch.eye(n,dtype=torch.float32).to(device)
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
            if num==0:
                num_class_dict[c]=0
                continue
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def train(self, verbose=True):#开始训练
        args = self.args
        data = self.data
        labels_syn = self.labels_syn#小图的X' A' Y'
        adj =  data.adj_full
        adj = utils.to_tensor1(adj, device='cpu')#先放在cpu
        syn_class_indices = self.syn_class_indices#小图输出分类对应的index

        # features=torch.FloatTensor(data.feat_full).to(self.device)
        # feat_sub, adj_sub = self.get_sub_adj_feat(features)#从大图X中选出对应的小图的X'
        # self.feat_syn.data.copy_(feat_sub)#小图的X'

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)#D^(-0.5)*(A+I)*D^(-0.5)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm#正则化之后的邻接矩阵A 2708*2708
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()#所有东西丢入深度学习网络之前都需要转变为tensor

        outer_loop, inner_loop = get_loops(args)#获得里外循环的超参数
        best_acc=0
        #获得了小图的X' 正则化后的A' Y',开始梯度下降
        print("开始图凝聚！")
        for it in range(args.epochs+1):
            print("epoch:",it)
            loss_avg = 0
            if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag']:#arxiv dropout，products不，papers dropout
                model = SGC1(nfeat=self.feat_syn.shape[1], nhid=self.args.hidden,
                            dropout=0, with_bn=False,
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

            perturb_adj = torch.FloatTensor(self.adj_syn.shape[0], self.adj_syn.shape[1]).uniform_(8e-3, 8e-3).to(self.device)#均匀抽样
            perturb_adj.requires_grad_()

            for ol in range(outer_loop):
                # adj_syn = self.pge(self.feat_syn)#PGE算法得到小图的邻接矩阵，小图不能太大，否则PGE的GPU内存会爆
                # adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                adj_syn_norm = self.adj_syn

                self.optimizer_feat.zero_grad()#X'梯度设置为0
                self.optimizer_pge.zero_grad()#φ梯度设置为0

                #Perturbation
                # for i in range(3):
                #     loss = torch.tensor(0.0).to(self.device)
                #     for c in range(data.nclass):#不同的类逐个分析
                #         if c not in self.num_class_dict:
                #             continue
                #         batch_size, n_id, adjs = data.retrieve_class_sampler(#得到大图中c类的节点和邻接矩阵 只有train中的节点
                #                 c, adj, transductive=True, args=args)             
                #         if args.nlayers == 1:
                #             adjs = [adjs]
                #         #大图
                #         feats_train=torch.FloatTensor(data.feat_full[n_id]).to(self.device)
                #         labels_train=data.labels_full[n_id[:batch_size]]
                #         if batch_size==1:
                #             labels_train=[labels_train]
                #         labels_train=torch.LongTensor(labels_train).to(self.device)
                #         adjs = [adj.to(self.device) for adj in adjs]
                #         output = model.forward_sampler(feats_train, adjs)
                #         loss_real = F.nll_loss(output, labels_train)
                #         gw_real = torch.autograd.grad(loss_real, model_parameters)#作用：计算并返回outputs对w的梯度
                #         gw_real = list((_.detach().clone() for _ in gw_real))#通过 tensor.detach().clone() 操作生成一个和原本 tensor 值相同的新 tensor 但是没有用到计算图所以节省了内存
                #         #小图
                #         output_syn = model.forward(self.feat_syn, adj_syn_norm+perturb_adj)
                #         ind = syn_class_indices[c]#c类的所有节点
                #         loss_syn = F.nll_loss(
                #                 output_syn[ind[0]: ind[1]],
                #                 labels_syn[ind[0]: ind[1]])
                #         gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                #         coeff = self.num_class_dict[c] / max(self.num_class_dict.values())#这个类loss的权重
                #         loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)/3#梯度匹配算法 
                    # #计算所有分类的梯度差和
                    # loss_avg += loss.item()#tensor的值
                    # loss.backward(retain_graph=True)
                    # perturb_data_adj = perturb_adj.detach() + 8e-3 * torch.sign(perturb_adj.grad.detach())#grad不会叠加，而optimizer会
                    # perturb_adj.data = perturb_data_adj.data
                    # perturb_adj.grad[:] = 0
                # self.optimizer_feat.step()

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):#不同的类逐个分析
                    if c not in self.num_class_dict:
                        continue

                    batch_size, n_id, adjs = data.retrieve_class_sampler(#得到大图中c类的节点和邻接矩阵 只有train中的节点
                            c, adj, transductive=True, args=args)
                            
                    if args.nlayers == 1:
                        adjs = [adjs]

                    #大图
                    feats_train=torch.FloatTensor(data.feat_full[n_id]).to(self.device)
                    labels_train=data.labels_full[n_id[:batch_size]]
                    if batch_size==1:
                        labels_train=[labels_train]
                    labels_train=torch.LongTensor(labels_train).to(self.device)
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(feats_train, adjs)
                    loss_real = F.nll_loss(output, labels_train)
                    gw_real = torch.autograd.grad(loss_real, model_parameters)#作用：计算并返回outputs对w的梯度
                    gw_real = list((_.detach().clone() for _ in gw_real))#通过 tensor.detach().clone() 操作生成一个和原本 tensor 值相同的新 tensor 但是没有用到计算图所以节省了内存

                    #小图
                    output_syn = model.forward(self.feat_syn, adj_syn_norm+perturb_adj)
                    ind = syn_class_indices[c]#c类的所有节点
                    loss_syn = F.nll_loss(
                            output_syn[ind[0]: ind[1]],
                            labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())#这个类loss的权重
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)#梯度匹配算法 
                loss_avg += loss.item()#tensor的值
                loss.backward()
                # if it % 50 < 10:
                #     self.optimizer_pge.step()
                # else:
                self.optimizer_feat.step()
                
                #不是直接对θ梯度下降，而是得到新的小图后，再代进去求新的loss然后梯度下降
                # feat_syn_inner = self.feat_syn.detach()
                # adj_syn_inner = self.pge.inference(feat_syn_inner)#φ x'-(pge)->a'
                # adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                # feat_syn_inner_norm = feat_syn_inner
                # for j in range(1):
                #     optimizer_model.zero_grad()
                #     output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                #     loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                #     loss_syn_inner.backward()
                #     optimizer_model.step() # update gnn param θ

                #直接用该模型作为训练模型
                # if ol%50==0:
                #     print("第",str(ol),"次梯度匹配+训练!")
                #     labels_test = torch.LongTensor(data.labels_test).cuda()
                #     labels_train = torch.LongTensor(data.labels_train).cuda()

                #     output = model.predict(data.feat_full, data.adj_full)
                #     loss_train = F.nll_loss(output[data.idx_train], labels_train)
                #     acc_train = utils.accuracy(output[data.idx_train], labels_train)
                #     if verbose:
                #         print("Train set results:",
                #               "loss= {:.4f}".format(loss_train.item()),
                #               "accuracy= {:.4f}".format(acc_train.item()))

                #     loss_test = F.nll_loss(output[data.idx_test], labels_test)
                #     acc_test = utils.accuracy(output[data.idx_test], labels_test)
                #     if verbose:
                #         print("Test set results:",
                #               "loss= {:.4f}".format(loss_test.item()),
                #               "accuracy= {:.4f}".format(acc_test.item()))

            #one epoch finished, the avg loss for every class in this epoch
            loss_avg /= (data.nclass*outer_loop)

            if verbose and it%100==0:

                # x = self.feat_syn.detach().cpu().numpy()
                # y = labels_syn.cpu().numpy()
                # n=y.max()+1
                # x_tsne = TSNE(n_components=2).fit_transform(x)

                # plt.figure(figsize=(8,8))
                # for i in range(n):
                #     plt.scatter(x_tsne[y == i, 0], x_tsne[y == i, 1])
                # plt.savefig(f'/home/xzb/GCond/pictures/{args.model}_{args.dataset}_{args.reduction_rate}_{it}.png')

                print('Epoch {}, 梯度匹配loss_avg: {}'.format(it, loss_avg))
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag'] else 3
                for i in range(runs):
                    if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag']:
                        res.append(self.test_with_val(data.feat_full, adj, best_acc))
                        best_acc=max(best_acc,res[0][1])
                    #多个runs的情况


    def test_with_val(self, features, adj, best_acc, verbose=True):#test    
        res = []

        data, device = self.data, self.device
        feat_syn, labels_syn = self.feat_syn.detach(), self.labels_syn

        if self.args.model=='GCN':
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=self.args.dropout,
                        weight_decay=5e-4, nlayers=2, with_bn=True,
                        nclass=data.nclass, device=device).to(device)
        else:
            model = DeeperGCN(self.args, nfeat=feat_syn.shape[1], nclass=data.nclass, device=device).to(device)

        # adj_syn = self.pge.inference(feat_syn)
        adj_syn = self.adj_syn
        args = self.args

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model.fit_with_val(args=args,features=feat_syn, adj=adj_syn, labels=labels_syn, data=data,train_iters=600, normalize=True, verbose=False, noval=True)
        model.eval()
        # Full graph
        labels_test = torch.LongTensor(data.labels_test).cuda()
        labels_train = torch.LongTensor(data.labels_train).cuda()

        output = model.predict(data.feat_full, data.adj_full)
        loss_train = F.nll_loss(output[data.idx_train], labels_train)
        acc_train = utils.accuracy(output[data.idx_train], labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())

        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))

        #subgraph
        # output = model.inference(features, adj)
        # loss_train = F.nll_loss(output[data.idx_train], torch.LongTensor(data.labels_train))
        # acc_train = utils.accuracy(output[data.idx_train], torch.LongTensor(data.labels_train))
        # res.append(acc_train.item())
        # if verbose:
        #     print("Train set results:",
        #           "loss= {:.4f}".format(loss_train.item()),
        #           "accuracy= {:.4f}".format(acc_train.item()))

        # loss_test = F.nll_loss(output[data.idx_test], torch.LongTensor(data.labels_test))
        # acc_test = utils.accuracy(output[data.idx_test], torch.LongTensor(data.labels_test))
        # res.append(acc_test.item())
        # if verbose:
        #     print("Test set results:",#put the full graph into the fitted gcn
        #           "loss= {:.4f}".format(loss_test.item()),
        #           "accuracy= {:.4f}".format(acc_test.item()))

        # if self.args.save and res[1]>best_acc:
        #     torch.save(adj_syn, f'/home/xzb/GCond/saved_ours/adj_{args.dataset}_{args.model}_{args.reduction_rate}_{args.seed}.pt')#save X' A'
        #     torch.save(feat_syn, f'/home/xzb/GCond/saved_ours/feat_{args.dataset}_{args.model}_{args.reduction_rate}_{args.seed}.pt')
        #     torch.save(model.state_dict(), f'/home/xzb/GCond/saved_model/{args.model}_{args.dataset}_{args.reduction_rate}_{args.seed}.pt') 

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
        k = 10
        sims = cosine_similarity(features.cpu().numpy())#计算小图中feature的两两距离
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0#对角线初始化为0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
            sims[i, indices_argsort[: -k]] = 0#除了k个相邻的（余弦最大的） 其他设置为0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn

def get_loops(args):
    if args.dataset in ['ogbn-arxiv','ogbn-products','ogbn-papers100M','ogbn-mag']:
        return 5, 0
    if args.dataset in ['cora']:
        return 20, 15 # sgc
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10

