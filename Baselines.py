import sys

from graph import Graph
from util import load_data, train_scheduler
import os.path as osp
import torch
import numpy as np
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.nn import SignedGCN

dataset_names = ['Bitcoin-OTC', 'Bitcoin-Alpha', 'WikiRfa', 'WikiElec', 'Epinions', 'Slashdot.bat']

for dataset_name in dataset_names:
    for i in range(1,6):
        name =  dataset_name + '-'+str(i)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        T =20
        initial = 0.25

        train, val, test = load_data(name)

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



        model = SignedGCN(64, 64, num_layers=2, lamb=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        x = x.to(device)

        res_best = {'val_auc': 0,'val_f1':0}
        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            z = model(x, train_pos_edge_index, train_neg_edge_index)
            loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
            loss.backward()
            optimizer.step()
            model.eval()
            res_cur = dict()
            with torch.no_grad():
                z = model(x, train_pos_edge_index, train_neg_edge_index)
                train_auc, train_f1 = model.test(z, train_pos_edge_index, train_neg_edge_index)
                val_auc, val_f1 = model.test(z, val_pos_edge_index, val_neg_edge_index)
                test_auc, test_f1 = model.test(z, test_pos_edge_index, test_neg_edge_index)
                res_cur['val_auc'] = val_auc
                res_cur['val_f1'] = val_f1
                if res_cur['val_auc'] +  res_cur['val_f1'] > res_best['val_auc'] + res_best['val_f1']:
                    res_best['val_auc'] =  res_cur['val_auc']
                    res_best['val_f1'] = res_cur['val_f1']
        print(f"dataset_name: {dataset_name},   "
              f"best auc: {res_best['val_auc']},    best f1: {res_best['val_f1']}")

                # print(f'epoch: {epoch:03d}, Loss: {loss:.4f} '
                #       f'train_auc: {train_auc:.4f}, train_f1: {train_f1:.4f} '
                #       f'val_auc: {val_auc:.4f}, val_f1: {val_f1:.4f} '
                #       f'test_auc: {test_auc:.4f}, test_f1: {test_f1:.4f} ' )


