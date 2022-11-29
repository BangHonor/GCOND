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


class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        n = int(len(data.idx_train) * args.reduction_rate)
        d = data.feat_train.shape[1]
        labels_syn=self.generate_labels_syn(data)
        self.labels_syn = torch.LongTensor(labels_syn).to(device)
        self.nnodes_syn = len(labels_syn)
        self.feat_syn = nn.Parameter(torch.FloatTensor(self.nnodes_syn, d).to(device))#由于生成标签的算法产生的label.shpe[0]可能会比原来的大，所以这里要对齐
        # self.pge = torch.eye(self.nnodes_syn,dtype=float).to(device)#单位矩阵

        x1=[]
        for i in range(self.nnodes_syn):
            x1.append(i)
        index=torch.LongTensor([x1,x1])
        val=torch.ones(torch.LongTensor(x1).shape,dtype=torch.float)
        self.pge=torch.sparse.FloatTensor(index, val, torch.Size([self.nnodes_syn,self.nnodes_syn])).to('cuda')
        
        if(self.labels_syn.shape[0]!=self.feat_syn.shape[0]):
            print("labels_syn与feat_syn规格不一致！")
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)


    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):#figure out!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        from collections import Counter
        counter = Counter(data.labels_train)#不一定有labels_full中的所有label
        counter_full=Counter(data.labels_full)
        print("labels_full:",counter_full)
        print("labels_train:",counter)

        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])#按照分类的节点数排序 class:num
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):#to make num of labels_syn=counter*redcution_rate 
            # if ix == len(sorted_counter) - 1:#最后一个多拿少补
            #     num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
            #     if num_class_dict[c]<1:
            #         num_class_dict[c]=1
            #     self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            #     labels_syn += [c] * num_class_dict[c]
            # else:
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict#记录小图中每个分类的节点数 大图train有的分类小图都有
        print("小图每个分类的节点数:",num_class_dict)
        return labels_syn

    def test_with_val(self,part_index,part,it,verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        args = self.args
        adj_syn=pge
        if args.save:
            torch.save(adj_syn, f'/home/xzb/GCond/saved_distillation/adj_{args.dataset}_{args.reduction_rate}_{args.seed}_{part_index}_{part}_{it}.pt')#store syn grpah
            torch.save(feat_syn, f'/home/xzb/GCond/saved_distillation/feat_{args.dataset}_{args.reduction_rate}_{args.seed}_{part_index}_{part}_{it}.pt')

        noval = True
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,#use syn graph, val data to train model
                     train_iters=600, normalize=True, verbose=False, noval=noval)
        model.eval()
        torch.save(model.state_dict(), f'/home/xzb/GCond/saved_distillation/GCN_{args.dataset}_{args.reduction_rate}_{args.seed}_{part_index}_{part}_{it}.pt') 
        print("存储GCN模型成功！")

        labels_test = torch.LongTensor(data.labels_test).cuda()

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
        if verbose:#output this iter's syn graph's best acc
            print("Test set results on syn graph's gnn:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))

        if False:
            if self.args.dataset == 'ogbn-arxiv':
                thresh = 0.6
            elif self.args.dataset == 'reddit':
                thresh = 0.91
            else:
                thresh = 0.7

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = utils.accuracy(output, labels_train)
            loss_train = torch.tensor(0)
            acc_train = torch.tensor(0)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
            res.append(acc_train.item())
        return res

    def train(self,part_index,part,verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn#syn是train*redcution_rate
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train#a partitioned graph
        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        # feat_sub, adj_sub = self.get_sub_adj_feat(features)#get original graphs's feature
        # self.feat_syn.data.copy_(feat_sub)

        #如果两者规格不同，说明有些类是缺失的，但是feat_sub考虑了所有的类，所以要改下get_sub_adj_feat

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()

        outer_loop, inner_loop = get_loops(args)
        maxacc=0

        #record the best syn graph
        best_syn_it=0
        best_acc=0
        for it in range(args.epochs+1):#1000训练小图
            loss_avg = 0
            if args.sgc==1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc==2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)

            else:
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                            device=self.device).to(self.device)

            model.initialize()#每一次循环都不一样

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):
                # adj_syn = pge
                # adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #GCN中是否有BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                for c in range(data.nclass):
                    if c not in self.num_class_dict:
                        #print("小图中没有",c,"这个分类，所以不去计算这个类的loss")
                        continue
                    batch_size, n_id, adjs = data.retrieve_class_sampler(#use big graph to get certain class's id and adj
                            c, adj, transductive=False, args=args)
                    #print(c,"分类计算完毕！")
                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))#big graph's grad, use it twice

                    adj_row=syn_class_indices[c][1]-syn_class_indices[c][0]
                    x1=torch.arange(start=0,end=adj_row,step=1)
                    index=torch.ones(2,adj_row)
                    index[0]=x1
                    index[1]=x1
                    val=torch.ones(x1.shape)
                    adj_syn=torch.sparse_coo_tensor(indices=index, values=val, size=[adj_row,adj_row]).to('cuda')

                    # if args.nlayers == 1:
                    #     adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    # else:
                    #     adj_syn_norm_list = [adj_syn_norm]*(args.nlayers-1) + \
                    #             [adj_syn_norm[ind[0]: ind[1]]]
                    ind = syn_class_indices[c]
                    output_syn = model.forward(feat_syn[ind[0]: ind[1]], adj_syn)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])

                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)#syn graph's grad
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                # # TODO: regularize
                # if args.alpha > 0:
                #     loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                # # else:
                # else:
                #     loss_reg = torch.tensor(0)

                # loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                loss.backward()
                self.optimizer_feat.step()

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break


                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=True)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step() # update gnn param

            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            eval_epochs = [0, 100, 200, 400, 600, 800, 1000]#最大1000，每当到了这些值就回去检验下此时的图训练出来的效果

            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 500 == 0:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv', 'reddit', 'flickr'] else 3
                for i in range(runs):
                    # self.test()
                    res.append(self.test_with_val(part_index,part,it))
                res = np.array(res)
                print('Test:',repr([res.mean(0), res.std(0)]))
                if(res.mean(0)>best_acc):
                    best_acc=res.mean(0)
                    best_syn_it=it
                # if(res.mean(0)>maxacc):
                #     maxacc=res.mean(0)
                # file=open("output.txt","a")
                # file.writelines(str(self.args.dataset)+'---epoch:'+str(it)+'---Train/Test Mean Accuracy:'+str(repr([res.mean(0), res.std(0)]))+'\n')
                # file.close()
        print("最好的得到小图的迭代轮次为",best_syn_it)
        return best_syn_it

    def get_sub_adj_feat(self, features):#get first feature from partitioned training feature and labels_syn  
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            if c not in self.num_class_dict:
                continue
            tmp = data.retrieve_class(c, num=counter[c])#choose num nodes's indices belong to c
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.dataset in ['ogbn-arxiv']:
        return 20, 0
    if args.dataset in ['reddit']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5 # at least 200 epochs
    else:
        return 20, 5

