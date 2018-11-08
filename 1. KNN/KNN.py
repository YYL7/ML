from __future__ import print_function
from numpy import *
import operator
from os import listdir
from collections import Counter

# create dataset and lables
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    inX: input 
    dataSet: training dataset
    labels: classes/lables
    k: K values
    kNN.classify0([0,0], group, labels, 3)
    """
    # 1. calculate the distance- Euclidean Distance
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # sqare it
    sqDiffMat = diffMat ** 2
    # add by row
    sqDistances = sqDiffMat.sum(axis=1)
    # square root
    distances = sqDistances ** 0.5
    # sort from small to large: argsort()
    sortedDistIndicies = distances.argsort()

    # 2. choose K nearest neigbors
    classCount = {}
    for i in range(k):
        # find the lables
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. sort and return the majority of the classfication
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
    
 def test1():
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))   
 
 # import training dataset   
 def file2matrix(filename):
    fr = open(filename)
    # length of the rows
    numberOfLines = len(fr.readlines())
    # form the matrix
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector   
    
  # data normalization
  # use min-max normalization
  def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    norm_dataset = (dataset - minvalue) / ranges
    return normDataSet, ranges, minVals
 
 # test by calculating the error rate
 def datingClassTest():
    # ratio of the testing set 
    hoRatio = 0.1  
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    # data normalization
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m: length of the rows
    m = normMat.shape[0]
    # numTestVecs: m smaple size
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # test
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)   
