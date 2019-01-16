from random import seed, randrange, random


# load DataSet 
def loadDataSet(filename):
    dataset = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            for featrue in line.split(','):
                # strip(),returns a copy of the string with both leading and trailing characters removed
                str_f = featrue.strip()
                if str_f.isdigit():   # Determine if it is a number
                    # Convert  to a float form
                    lineArr.append(float(str_f))
                else:
                    # append with features
                    lineArr.append(str_f)
            dataset.append(lineArr)
    return dataset

#cross validation--split
def cross_validation_split(dataset, n_folds):
    """cross_validation_split(resampled by n_folds repeatedly. Each time the elements of the list are non-repeating.)
    Args:
        dataset:     dataset
        n_folds:     resampled by n_folds
    Returns:
        dataset_split:    list：dataset with n_folds 
    """
    dataset_split = list()
    dataset_copy = list(dataset)       # copy in case of any change
    fold_size = len(dataset) / n_folds
    for i in range(n_folds):
        fold = list()                  # Clear each time fold fold to prevent duplicate 
        while len(fold) < fold_size:              
            index = randrange(len(dataset_copy))
            # Export the contents of the corresponding index from dataset_copy and remove the content from dataset_copy.
            # pop(), remove an element from the list (the default last element) and return the list.
            # fold.append(dataset_copy.pop(index))  # No return
            fold.append(dataset_copy[index])  # return
        dataset_split.append(fold)
    # dataset with n_folds
    return dataset_split


# Split a dataset based on an attribute and an attribute value 
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):    # the more accurate the classification, the smaller the gini
    gini = 0.0
    for class_value in class_values:     # class_values = [0, 1] 
        for group in groups:             # groups = (left, right)
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))   
    return gini


# find the best feature to split, and get the index, row[index],and groups（left, right）
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))  # class_values =[0, 1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)  
        if index not in features:
            features.append(index)
    for index in features:                   
        for row in dataset:
            groups = test_split(index, row[index], dataset)  
            gini = gini_index(groups, class_values)
            # The number of the left and right sides is the same, indicating that not much difference, thus we have larger gini. 
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups  # get the best feature to split
    # print(b_score)
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value 
# return a label with maximum occurrences in the group
def to_terminal(group):
    outcomes = [row[-1] for row in group]           
    return max(set(outcomes), key=outcomes.count)   


# Create child splits for a node or make terminal  
def split(node, max_depth, min_size, n_features, depth):  # max_depth = 10, min_size = 1, n_features = int(sqrt((dataset[0])-1))
    left, right = node['groups']
    del(node['groups'])
# check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
# check for max depth
    if depth >= max_depth:  
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
# process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features) 
        split(node['left'], max_depth, min_size, n_features, depth+1)  
# process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    """build_tree
    Args:
        train:           training dataset
        max_depth:       max depth of the tree
        min_size:        size of the leaf node
        n_features:      number of the features
    Returns:
        root             return the tree
    """
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):   
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):       # isinstance(), checks if the object (first argument) is an instance or subclass.
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    """bagging_predict
    Args:
        trees:           list of the trees
        row:             row data of the training dataset
    Returns:
        Return the majority of the random forest
    """

    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):  
    """random_forest
    Args:
        dataset         training dataset
        ratio           ratio for the training dataset
    Returns:
        sample          random sample for the training dataset
    """

    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        # sample with return
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """random_forest
    Args:
        train:           training dataset
        test:            testing dataset
        max_depth:       maximum depth of the tree
        min_size:        size of the leaf node
        sample_size:     sample size of the training set
        n_trees:         number of trees
        n_features:      number of the features
    Returns:
        predictions:     predictions for each row
    """

    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        # buile a tree
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    # predictions for each row
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):  # input the actual and predicted value, get the accuracy)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# evaluate the algorithm
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """evaluate_algorithm
    Args:
        dataset:     original dataset
        algorithm:   algorithm for using
    Returns:
        scores:      scores for the algorithms
    """

    # resample by n_folds
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        # fold, represents the test set extracted from the original dataset 
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        # calculate the accuracy
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == '__main__':

    # load the dataset
    dataset = loadDataSet('6.RandomForest/sonar-all-data.txt')
    # print(dataset)

    n_folds = 5        
    max_depth = 20    
    min_size = 1      
    sample_size = 1.0 
    # n_features = int((len(dataset[0])-1))
    n_features = 15     
    for n_trees in [1, 10, 20]:  
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
