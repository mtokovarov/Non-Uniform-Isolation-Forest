import numpy as np
from node import Node
from tree import Tree


class TreeGrowerBasic:
    def __init__(self, X, sample_size, DataPreparator = None):
        self.X = X
        self.dim_cnt = self.X.shape[1]
        self.sample_size = sample_size
        self.indeces = np.arange(0, self.X.shape[0])
        self.DataPreparator = DataPreparator
        self.X_samples = []
    
    def make_datasets(self, ds_cnt):
        for i in range(ds_cnt):
            self.X_samples.append(self.make_sample_dataset())
    
    def make_tree(self, limit = None):
        if (limit is None):
            limit = int(np.ceil(np.log2(self.sample_size)))
        self.grown_tree = Tree()
        X_sample = self.make_sample_dataset()
        if (self.DataPreparator is not None):
            X_sample = self.DataPreparator.prepare_data(X_sample)
        self.X_samples.append(X_sample)
        self.grown_tree.root = self.recursively_grow(X_sample, 0, limit) 
        return self.grown_tree
    
    def transform_data_with_preparator(self):
        for i in range(len(self.X_samples)):
            self.X_samples[i] = self.DataPreparator.prepare_data(self.X_samples[i])
    
    def regrow_trees(self, limit = None):
        if (limit is None):
            limit = int(np.ceil(np.log2(self.sample_size)))
        trees = []
        for i in range(len(self.X_samples)):
            self.grown_tree = Tree()
            self.grown_tree.root = self.recursively_grow(self.X_samples[i], 0, limit) 
            trees.append(self.grown_tree)
        return trees
    
    def recursively_grow(self, X, tree_depth,depth_limit):
        self.grown_tree.tree_depth = tree_depth
        if tree_depth >= depth_limit or len(X) <= 1:
            self.grown_tree.exnodes += 1
            return Node(X, self.grown_tree.rot_op, self.grown_tree.border, tree_depth, 
                        left = None, right = None, node_type = 'exNode' )
        else:
            self.grown_tree.rot_op = self.get_rot_operator(X)
            X_rot = self.get_rotated(X, self.grown_tree.rot_op)
            if X_rot.min()==X_rot.max():
                self.grown_tree.exnodes += 1
                return Node(X, self.grown_tree.rot_op, self.grown_tree.border, 
                        tree_depth, left = None, right = None, node_type = 'exNode' )
            self.grown_tree.border = self.get_border(X_rot)
            w = np.where(X_rot < self.grown_tree.border,True,False)
            return Node(X, self.grown_tree.rot_op, self.grown_tree.border, tree_depth,\
            left=self.recursively_grow(X[w,:], tree_depth+1, depth_limit),\
            right=self.recursively_grow(X[~w,:],tree_depth+1,depth_limit),\
            node_type = 'inNode' )   
                
    def get_rotated(self, X, rot_op):
        return np.dot(X, rot_op)
    
    def make_sample_dataset(self):
        selected_indeces = np.random.choice(self.indeces, self.sample_size, replace = False)
        return self.X[selected_indeces,:]
        
    #methods to be redefined in the child classes - we can modify the axis selection and split value generation
    
    def get_rot_operator(self, X):
        index = np.random.choice(np.arange(0, X.shape[1]))
        rot_op = np.zeros(X.shape[1])
        rot_op[index] = 1
        return rot_op
        
    def get_border(self, X):
        min_val = min(X)
        max_val = max(X)
        return np.random.rand()*(max_val - min_val) + min_val
        