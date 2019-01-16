# Random Forest

Random Forest is an Ensemble method. Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.

Firstly, randomly choose data, to build N sub-datasets by randomly selecting data from the original dataset N times with replacement. The sub-datasets have the same size as the original.  And then we build the decision tree for each sub-dataset, and each tree will give us a class for given a new data. So, the majority will be the new class for the new data.  

Secondly, randomly choose the best features to split.
