# Decision Tree

from __future__ import print_function
print(__doc__)
import operator
from math import log
from collections import Counter




# measure the entropy, split the dataset, measure the
# entropy on the split sets, and see if splitting it was the right thing to do.

#  use createDataSet() to input data
def createDataSet():
    """
    DateSet 
    Args:
        No need to input features
    Returns:
        dataSet, labels
    """
    # dataSet = [['yes'],
    #         ['yes'],
    #         ['no'],
    #         ['no'],
    #         ['no']]
    # labels:  no surfacing & flippers
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels
  
  
  

# calculate the entropy
def calcShannonEnt(dataSet):
    """calcShannonEnt(calculate Shannon entropy )
    Args:
        dataSet 
    Returns:
        For each attribute/feature, rertun the entropy by each class
    """
    # length of the list, which represent the size of the training dataset
    numEntries = len(dataSet)
    # to count the times of each label 
    labelCounts = {}
    # the the number of unique elements and their occurance
    for featVec in dataSet:
        # to store the labels for our exsiting data, the end of each row represent the label
        currentLabel = featVec[-1]
        # create the dictionary for each lable
        # if the key was not encoutered previously, one is created.
        # for each key, keep track of the how many times this label occur. [Key: Value]: [label: frequency of label]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        # print '-----', featVec, labelCounts

    # calculate the entropy for labels by ratios of labels
    shannonEnt = 0.0
    for key in labelCounts:
        # use the frequency of all differernt labels to calculate the probability of that label.
        prob = float(labelCounts[key])/numEntries
        # log base 2 
        shannonEnt -= prob * log(prob, 2)
        # print ('---', prob, prob * log(prob, 2), shannonEnt)
    return shannonEnt

## second way to calculate entropy
## count the numbers of labels
#label_count = Counter(data[-1] for data in dataSet)
## calculate the probability
#probs = [p[1] / len(dataSet) for p in label_count.items()]
## calculate the entropy
#shannonEnt = sum([-p * log(p, 2) for p in probs])      
  
  
  
 
# Create the separate list 
def splitDataSet(dataSet, index, value):
    """
    Desc：
        splitDataSet
    Args:
        dataSet  --  the dataset we will split
        index --  the feature we’ll split on
        value --  the value of the feature to return 
    Returns:
        dateset which chop out the index used for splitting
    """
    retDataSet = []
    for featVec in dataSet: 
        if featVec[index] == value:
            # Cut out the feature split on
            # Chop out index used for splitting
            # [:index]: take the items before the specific index
            # [index+1:]: skip the row of index+1，take the remaining data
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

##Second way to split the data
# retDataSet = [data[:index] + data[index + 1:] for data in dataSet for i, v in enumerate(data) if i == index and v == value]





# For all of our features to determine the best feature to split on
def chooseBestFeatureToSplit(dataSet):
    """
    Desc:
        determine the best feature to split on
    Args:
        dataSet -- the dataset we’ll split
    Returns:
        bestFeature -- best feature to split on
    """
    # number of features for the first row (last one is the label, so we minus one)
    numFeatures = len(dataSet[0]) - 1
    # Entropy for the label
    baseEntropy = calcShannonEnt(dataSet)
    # best infomation gain, best feature 
    bestInfoGain, bestFeature = 0.0, -1
    # iterate over all the features
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        # make a list to get the i+1 feature for each instance/example 
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        # set: to remove duplicate
        uniqueVals = set(featList)
        
        # Calculate entropy for each split
        # create a new entropy
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # infomation gain: The change of information before and after the split.
        # The split with highest information gain is the best option.
        # Lastly，compare the information gain for all the features to find the best info gain
        # so that we get our best feature with best info gain
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# # second way to find the best features
# # calculate the base entropy before spliiting the dataset
# base_entropy = calcShannonEnt(dataSet)
# best_info_gain = 0
# best_feature = -1
# # for each feature
# for i in range(len(dataSet[0]) - 1):
#     # count the current feature
#     feature_count = Counter([data[i] for data in dataSet])
#     # calculate the new entropyb(after ssplitting)
#     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
#                    for feature in feature_count.items())
#     # information gain
#     info_gain = base_entropy - new_entropy
#     print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
#     if info_gain > best_info_gain:
#         best_info_gain = info_gain
#         best_feature = i
# return best_feature





# This function takes a list of class names and then creates a dictionary, 
# whose keys are the unique values in classList, and the object of the dictionary is the
# frequency of occurrence of each class label from classList. [key:value]:[label: frequency]
# Finally, you use the  operator to sort the dictionary by the keys and return the class that occurs with the greatest frequency
def majorityCnt(classList):
    """
    Desc:
        choose the majority 
    Args:
        classList (list of the lalel or class column)
    Returns:
        bestFeature 
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # reverse order, the first one of the dictionary is the majority 
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('sortedClassCount:', sortedClassCount)
    return sortedClassCount[0][0]


## Second way to count the majority
# major_label = Counter(classList).most_common(1)[0]
# return major_label





# create tree
def createTree(dataSet, labels):
    """
    Desc:
        createTree
    Args:
        dataSet -- training dataset for building the tree
        labels -- labels in the training dataset
    Returns:
        myTree -- finished tree
    """
    # [-1]: last one, the label
    classList = [example[-1] for example in dataSet]
    # If the count of the first value of the label column, equal the length of the dataset
    # then we have only one label, so return the first value of the label column
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # When no more features,return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # best feature
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # get the name
    bestFeatLabel = labels[bestFeat]
    # initialize myTree
    myTree = {bestFeatLabel: {}}

    del(labels[bestFeat])
    # choose the best feature，use it's branch for classification
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # remaining label
        subLabels = labels[:]
        # for the different value of the current label
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print('myTree', value, myTree)
    return myTree

  
  
# Testing: Perform classification for the new dataset using decision tree
def classify(inputTree, featLabels, testVec):
    """
    Desc:
        classification for the new dataset
    Args:
        inputTree  -- trained tree model
        featLabels -- The name corresponding to the Feature, not the target variable
        testVec    -- Testing dataset
    Returns:
        classLabel -- classfication result
    """
    # get the key value of the tree root
    firstStr = list(inputTree.keys())[0]
    # value of the tree root(firstStr)
    secondDict = inputTree[firstStr]
    # Translate label string to index
    featIndex = featLabels.index(firstStr)
    # Testing dataset
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # Determine if the branch is over: By Determining if valueOfFeat is a dict type
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel
  


  
  
  
# store the tree model
def storeTree(inputTree, filename):
    """
    Desc:
        store the tree model built by importin pickle
    Args:
        inputTree -- trained model
        filename -- name to store the tree
    Returns:
        None
    """
    import pickle
    
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# second way to store the tree
#   with open(filename, 'wb') as fw:
#       pickle.dump(inputTree, fw)




def grabTree(filename):
    """
    Desc:
        retrieve the tree built by importing pickle
    Args:
        filename -- the name we use to store the tree
    Returns:
        pickle.load(fr) -- retrieve the tree built
    """
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
  
