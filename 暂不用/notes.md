class GCond:
    
    def __init__(self, data, args, device='cuda', **kwargs):
        #把传进来的参数赋予给self
        self.data = data
        self.args = args
        self.device = device
        
        #确定模型的基本结构、参数：输入 模型要用到的子模型及其参数初始化 输出 
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)
        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)


    def test_with_val(self, verbose=True):
        #用val数据进行fit
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data, train_iters=600, normalize=True, verbose=False)
        model.eval()

        #用train和test数据到fit过后的模型
        labels_test = torch.LongTensor(data.labels_test).cuda()
        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)#140*1433 
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        print("Train set results from training nodes:",#put training data into the fitted gcn
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        return res

    def train(self, verbose=True):
        #将所有输入、输出、参数化为tensor
        ...
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        #稀疏矩阵的处理
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)#D^(-0.5)*(A+I)*D^0.5
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()

        #开始梯度下降
        for it in range(args.epochs+1):
            #初始化子模型，优化器，训练之前model.train()
            model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                        nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                        device=self.device).to(self.device)
            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            #算法主题思路
            #如何实现整个模型的train forward，给出loss的表达式，梯度下降
            optimizer_model.zero_grad()
            output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
            loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
            loss_syn_inner.backward()
            optimizer_model.step() 

            #到达一定epoch时可以用val data进行fit
            eval_epochs = [10,50,400, 600, 800, 1000, 1200, 1600, 2000]
            if verbose and it in eval_epochs:
                self.test_with_val()
            


