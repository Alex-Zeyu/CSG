import networkx as nx
import numpy as np
import csv
import sys
import torch


class Graph(object):
    def __init__(self, edges, directed=False):
        self.edges = edges
        positive_graph = nx.DiGraph() if directed else nx.Graph()
        negative_graph = nx.DiGraph() if directed else nx.Graph()
        for i, j, w in edges:
            if w >0:
                positive_graph.add_edge(i,j)
            if w <0:
                negative_graph.add_edge(i,j)

        self.positive_graph = positive_graph
        self.negative_graph = negative_graph

    def get_positive_edges(self):
        return self.positive_graph.edges()

    def get_negative_edges(self):
        return self.negative_graph.edges()

    def __len__(self):
        return max(len(self.positive_graph), len(self.negative_graph))


    def edge_score(self):
        score = dict()
        for i, j, w in self.edges:
            balance = 0 # balanced number
            unbalance = 0 #
            i_pos = None
            i_neg = None
            j_pos = None
            j_neg = None
            if i in self.positive_graph:
                i_pos = set(self.positive_graph[i])
            if i in self.negative_graph:
                i_neg = set(self.negative_graph[i])
            if j in self.positive_graph:
                j_pos =  set(self.positive_graph[j])
            if j in self.negative_graph:
                j_neg = set(self.negative_graph[j])

            pos_pos_n = len(i_pos & j_pos) if (i_pos is not None and j_pos is not None) else 0
            pos_neg_n = len(i_pos & j_neg) if (i_pos is not None and j_neg is not None) else 0
            neg_pos_n = len(i_neg & j_pos) if (i_neg is not None and j_pos is not None) else 0
            neg_neg_n = len(i_neg & j_neg) if (i_neg is not None and j_neg is not None) else 0

            if w>0:
                balance = pos_pos_n + neg_neg_n
                unbalance = pos_neg_n + neg_pos_n
            if w<0:
                balance = pos_neg_n + neg_pos_n
                unbalance = pos_neg_n + neg_pos_n
            degree = 1-balance/(balance + unbalance) if (balance + unbalance)!=0 else 0
            score[(i,j,w)] = degree
        return score





