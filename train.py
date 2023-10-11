import sys

from graph import Graph
from util import load_data, train_scheduler
import os.path as osp
import torch
import numpy as np
from torch_geometric.nn import SignedGCN
import torch_geometric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='Bitcoin-OTC-1',
                    help='choose dataset')
parser.add_argument('--seed', type=int, default=2023,
                    help='Random seed.')
parser.add_argument('--split', type=str, default='linear',
                    choices=['linear', 'root', 'geometric'],
                    help='pacing function')
parser.add_argument('--lambda_0', type=float, default=0.25,
                    help='initial proportion')
parser.add_argument('--T', type=int, default=20,
                    help='Split group')
args = parser.parse_args()
print(args)

torch_geometric.seed_everything(args.seed)
dataset_name =args.dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T =args.T
initial = args.lambda_0

train, val, test = load_data(dataset_name)

node_count = (np.concatenate([train, val, test], axis=0))[:,0:2].max() +1

train_pos_mask = train[:,2]>0
train_neg_mask = train[:,2]<0
val_pos_mask = val[:,2]>0
val_neg_mask = val[:,2]<0
test_pos_mask = test[:,2]>0
test_neg_mask = test[:,2]<0

train_pos_edge_index = torch.from_numpy(train[train_pos_mask ,0:2].T).to(device).long()
train_neg_edge_index = torch.from_numpy(train[train_neg_mask ,0:2].T).to(device).long()

val_pos_edge_index = torch.from_numpy(val[val_pos_mask ,0:2].T).to(device).long()
val_neg_edge_index = torch.from_numpy(val[val_neg_mask ,0:2].T).to(device).long()

test_pos_edge_index = torch.from_numpy(test[test_pos_mask ,0:2].T).to(device).long()
test_neg_edge_index = torch.from_numpy(test[test_neg_mask ,0:2].T).to(device).long()

x = torch.from_numpy(np.random.rand(node_count, 64).astype(np.float32))


train_graph = Graph(train)

score =train_graph.edge_score()

score = dict(sorted(score.items(), key=lambda x: x[1]))

score_edges = np.array(list(score.keys()))

score_edges = torch.from_numpy(score_edges).to(device).long()

scheduler = train_scheduler(initial=initial, T=T, method=args.split)


model = SignedGCN(64, 64, num_layers=2, lamb=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
x = x.to(device)

edge_n = train.shape[0]

res_best = {'val_auc': 0,'val_f1':0}
for id, s in enumerate(scheduler):
    for epoch in range(101):
        temp_n = int( edge_n * s)
        if id <= T:
            temp_edges = score_edges[:temp_n]
        else:
            temp_edges = score_edges
        temp_pos_mask = temp_edges[:,2]>0
        temp_neg_mask = temp_edges[:,2]<0
        temp_pos_edge = temp_edges[temp_pos_mask, 0:2].T
        temp_neg_edge = temp_edges[temp_neg_mask, 0:2].T
        model.train()
        optimizer.zero_grad()
        z = model(x, train_pos_edge_index, train_neg_edge_index)
        loss = model.loss(z, temp_pos_edge, temp_neg_edge)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            z = model(x, train_pos_edge_index, train_neg_edge_index)
        train_auc, train_f1 = model.test(z, train_pos_edge_index, train_neg_edge_index)
        val_auc, val_f1 = model.test(z, val_pos_edge_index, val_neg_edge_index)
        test_auc, test_f1 = model.test(z, test_pos_edge_index, test_neg_edge_index)
        res_cur = dict()
        res_cur['val_auc'] = val_auc
        res_cur['val_f1'] = val_f1
        if res_cur['val_auc'] + res_cur['val_f1'] > res_best['val_auc'] + res_best['val_f1']:
            res_best['val_auc'] = res_cur['val_auc']
            res_best['val_f1'] = res_cur['val_f1']
        # print(f'group: {id:03d}, epoch: {epoch:03d}, Loss: {loss:.4f} '
        #       f'train_auc: {train_auc:.4f}, train_f1: {train_f1:.4f} '
        #       f'val_auc: {val_auc:.4f}, val_f1: {val_f1:.4f} '
        #       f'test_auc: {test_auc:.4f}, test_f1: {test_f1:.4f} ' )
print(f"dataset_name: {dataset_name},   best auc: {res_best['val_auc']},    best f1: {res_best['val_f1']}")


