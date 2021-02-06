#The author used original code of Matias Carrasco Kind as the basis for further development
__author__ = 'Mikhail Tokovarov'
import numpy as np
import warnings
from path_factor import PathFactor
import os
from tree_grower_basic import TreeGrowerBasic
clear = lambda: os.system('cls')

try:
    import igraph as ig
except:
    warnings.warn("No igraph interface for plotting trees")
    

def c_factor(n) :
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

class IsolationForest():
    def __init__(self,treeGrower, X, tree_cnt,  sample_size, limit=None):
        self.tree_cnt = tree_cnt
        self.X = X
        self.sample_size = sample_size
        self.Trees = []
        self.c = c_factor(self.sample_size)  
        self.treeGrower = treeGrower
        self.limit = limit
    
    def grow_forest(self):
        for i in range(self.tree_cnt):
            self.Trees.append(self.treeGrower.make_tree(self.limit))
            
    def regrow_forest(self):
        self.Trees = self.treeGrower.regrow_trees(self.limit)
        
    def compute_paths(self, X_in = None):
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in  range(len(X_in)):
            h_temp = 0
            for j in range(self.tree_cnt):
                h_temp += PathFactor(X_in[i],self.Trees[j]).path*1.0
            Eh = h_temp/self.tree_cnt
            S[i] = 2.0**(-Eh/self.c)
        return S

    def compute_paths_single(self, x):
        S = np.zeros(self.tree_cnt)
        for j in range(self.tree_cnt):
            path =  PathFactor(x,self.Trees[j]).path*1.0
            S[j] = 2.0**(-1.0*path/self.c)
        return S

def make_graph(tree):
    g=ig.Graph()
    counter = [0]
    def recursively_make_graph(node, counter, g, parent_id=0):
        g[0].add_vertex(counter[0])
        current_id = counter[0]
        if (current_id != 0):
            g[0].add_edge(current_id, parent_id)
        counter[0] += 1
        if (node.left is not None): recursively_make_graph(node.left, counter, g, current_id)
        if (node.right is not None):recursively_make_graph(node.right, counter, g, current_id)    
    recursively_make_graph(tree.root, counter, [g])
    g.vs["label"] = list(range(counter[0]))
    return g


