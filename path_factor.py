import numpy as np

def c_factor(n) :
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


class PathFactor(object):
    def __init__(self,x,itree):
        self.path_list=[]        
        self.x = x
        self.e = 0
        self.path = self.find_path(itree.root)

    def find_path(self,T):
        if T.ntype == 'exNode':
            if T.size == 1: return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            xa = np.dot(self.x, T.rot_op)
                    
            self.e += 1
            if xa <= T.border:
                self.path_list.append('L')
                return self.find_path(T.left)
            else:
                self.path_list.append('R')
                return self.find_path(T.right)