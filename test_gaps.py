import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from generatingDatasets import DatasetGenerator
from isolation_forest import IsolationForest
from tree_grower_gaps_split import TreeGrowerGapsSplit
from tree_grower_basic import TreeGrowerBasic


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


growers = [TreeGrowerGapsSplit, TreeGrowerBasic]
grower_names = [grower.__name__ for grower in growers]
grower_dict = {grower.__name__:grower for grower in growers}

cluster_cnt = 5
dim_cnts = [2, 3, 4, 5, 6, 7, 8, 9, ] # 
total_sample_cnt = 5000
outlier_perc = 1
min_radius = 0.1
max_radius = 0.4
shuffle = True
closeness_tolerance = 0.05
overlapping  = False

tree_cnt = 100
sample_size = 256
repeat_cnt = 100

#make sure you have the following folders: 'gap_grower' and 'gap_grower\\generated_data'
path = 'gap_grower'
data_path = f'{path}\\generated_data'
for j, dim_cnt in enumerate(dim_cnts):
    scores = {grower_name:np.zeros((repeat_cnt, total_sample_cnt)) for grower_name in grower_dict.keys()}
    dg = DatasetGenerator(cluster_cnt, dim_cnt, total_sample_cnt, outlier_perc,
                              min_radius, max_radius, shuffle, closeness_tolerance,
                              overlapping = overlapping)
        
    data = np.zeros((repeat_cnt, total_sample_cnt, dim_cnt))
    labels = np.zeros((repeat_cnt, total_sample_cnt))
    
    for i in range(repeat_cnt):
        dg.generate_data()
        X, y = dg.get_data()
        data[i,...] = X
        labels[i,...] = y
        print('generated')
        for grower_name, grower in grower_dict.items():
            print('repeat num ', i, ' dims: ', dim_cnt, ' grower: ',  grower_name)
            new_grower = grower(X,sample_size)
            forest = IsolationForest(new_grower, X, tree_cnt, sample_size)
            forest.grow_forest()
            scores[grower_name][i,...] = forest.compute_paths()
            print(f"results are ready for {grower_name}!:", roc_auc_score(y, scores[grower_name][i,...]))
    file_name = f'{path}/dependent_sample_{total_sample_cnt}_sample_{dim_cnt}_dims.pkl'
    save_dict(scores, file_name)
    data_file_path = f'{data_path}\\new_gaps_growers_data_{dim_cnt}_dims.npz'
    np.savez(data_file_path, labels = labels, data = data)
