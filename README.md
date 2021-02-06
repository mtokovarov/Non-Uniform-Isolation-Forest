# Non-Uniform-Isolation-Forest
The code implementing Non_Uniform extension of Isolation Forest algorithm.
If you use the code, please cite the paper [paper details].

The code is based upon the original Isolation Forest implementation by Matias Carrasco Kind: https://github.com/mgckind/iso_forest

The structure underwent the following changes:
  -The field names in the classes were changed to more meaningful
  
  -The procedure of building an isolation tree was separated into standalone class named "TreeGrowerBasic"
  
  -All the modifications in the procedure of building an isolation tree are to be done in the form of child classes of "TreeGrowerBasic" overriding respective methods: 
      
      get_rot_operator - selection of axis to perform a split along. The method returns 'rot_operator' - a vector of the length equal to the number of dimensions in the analyzed dataset. The dataset is cast onto selected axis by dot product with 'rot_operator'.
      
      get_border - selection of the split value - in TreeGrowerBasic it is uniformly generated random value between Xmin and Xmax.

The class TreeGrowerGapsSplit overrides the method 'get_border' by the parent class ensuring the piecewise probability density function proportional to the length of the intervals between neighboring datapoints.

An instance of TreeGrower has to be passed to the constructor of IsolationForest. The type of TreeGrower defines the type of Isolation Forest.

An instance of TreeGrowerBasic can have a DataPreparator - a class implementing the method "prepare_data" - can be potentially used for training data preprocessing. Default value is None - no preprocessing applied.
