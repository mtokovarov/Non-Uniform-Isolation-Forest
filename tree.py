class Tree(object):
    def __init__(self):
        self.tree_depth = 0
        self.border = None
        self.rot_op = None
        self.exnodes = 0
        self.root = None
        
    def get_node(self, path):
        node = self.root
        for p in path:
            if p == 'L' : node = node.left
            if p == 'R' : node = node.right
        return node