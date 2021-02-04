from tree_grower_basic import TreeGrowerBasic
import numpy as np

def generateTwoSidedRandom(left, right):
    rand_val = ((np.random.rand() - np.random.rand()) + 1)*0.5
    return left + rand_val*(right - left)

class TreeGrowerGapsSplit(TreeGrowerBasic):   
    def get_border(self, X):
        p = generateTwoSidedRandom(0,1)
        X_sorted = np.unique(X)
        dists = np.diff(X_sorted)
        k = 1/(dists**2).sum()
        i=0
        cumulated = X_sorted[0]+np.spacing(X_sorted[0])
        while(dists[i]**2*k<p):
            p -= dists[i]**2*k
            cumulated += dists[i]
            i+=1
        border = cumulated + p/(dists[i]*k)
        return border
        