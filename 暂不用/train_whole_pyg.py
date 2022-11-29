import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from models_pyg.gcn import GCN
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

parser = argparse.ArgumentParser(description='ogbn-arxiv (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
args = parser.parse_args()
print(args)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                    transform=T.ToSparseTensor())

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
# val_idx = split_idx['valid'].to(device)
# idx=torch.cat([train_idx,val_idx],dim=0) 

if args.model=='SAGE':
    model = SAGE(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
else:
    model = GCN(data.num_features, args.hidden_channels,
                dataset.num_classes, args.num_layers,
                args.dropout).to(device)

evaluator = Evaluator(name='ogbn-arxiv')
best_acc=0
best_model=model

model.reset_parameters()
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    for epoch in range(1, 1 + args.epochs):
        optimizer.zero_grad()
        out = model(data.x[train_idx], data.adj_t[train_idx,train_idx])

        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        loss = loss.item()
        result = test(model, data, split_idx, evaluator)

        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
    if valid_acc>best_acc:
        valid_acc=best_acc
        best_model=model

result = test(model, data, split_idx, evaluator)
train_acc, valid_acc, test_acc = result
print(  f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')
torch.save(best_model.state_dict(), f'/home/xzb/GCond/saved_distillation/{args.model}_Whole_{args.dataset}_{args.seed}.pt') 

