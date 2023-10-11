import math
import sys
import os
import numpy as np
import networkx as nx
import torch

def load_data(dataset_name):
    train_file_path = os.path.join('datasets', f'{dataset_name}_train.txt')
    val_file_path = os.path.join('datasets', f'{dataset_name}_val.txt')
    test_file_path = os.path.join('datasets', f'{dataset_name}_test.txt')

    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)


def train_scheduler(initial=0.25, T =50, method='linear'):
    ratio = []
    ratio.append(initial)
    if method == 'linear':
        for t in range(1,T+1):
            temp = (1- initial) * t / T + initial
            ratio.append(temp)
    elif method== 'root':
        for t in range(1, T+1):
            temp = (1 - initial**2) * t /T
            temp = math.sqrt(temp + initial**2)
            ratio.append(temp)
    elif method == 'geometric':
        for t in range(1, T+1):
            temp = math.log2(initial) -  math.log2(initial)*t / T
            temp = math.pow(2,temp)
            ratio.append(temp)
    return ratio

