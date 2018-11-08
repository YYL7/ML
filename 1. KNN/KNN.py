# import package
from __future__ import print_function
from numpy import *
import operator
from os import listdir
from collections import Counter

# create dataset and lables
def createDataSet():
    """
    group -- attributes/features on the training dataset
    lables -- lables of the training dataset
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    inX--input vector/training dataset
    dataSet--features/attributes of the training dataset
    labels--classes/lables of the training dataset
    k--K values
    to predict the lables-- kNN.classify0([0,0], group, labels, 3)
    """
    # 1. calculate the distance- Euclidean Distance ()
    dataSetSize = dataSet.shape[0]
    # tile --form the same matrix to the training dataset, then calculate the difference
    # tile(inX,(3,1)), 3 means the rows copyed, 1 means the copy times
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # square the matricx difference 
    sqDiffMat = diffMat ** 2
    # add all squared distances
    sqDistances = sqDiffMat.sum(axis=1)
    # set square root
    distances = sqDistances ** 0.5
    # sort from small to large: argsort()
    sortedDistIndicies = distances.argsort()

    # 2. choose K nearest neigbors
    classCount = {}
    for i in range(k):
        # find the lables
        voteIlabel = labels[sortedDistIndicies[i]]
        # get() -- if we get specific lable, then we add 1 to that lable; else, we add 0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
    # 3. sort and return the majority of the classfication
    # items() -- return the list, which have key and value in a tuple, {(key, value),(key, value)}
    # key=operator.itemgetter(1) -- sort the second element first, which is the lable in this case
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# senond way to do classify0():
#def classify0():
#    # 1. calculate the distance- Euclidean Distance ()
#    dist = np.sum((inx - dataset)**2, axis=1)**0.5
#    # 2. choose K nearest neigbors
#    k_labels = [labels[index] for index in dist.argsort()[0 : k]]
#    # 3. the majority will be the lable to the new data
#    label = Counter(k_labels).most_common(1)[0][0]
#    return label

# first example
 def test1():
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))   
 
 # import the training dataset   
 def file2matrix(filename):
    fr = open(filename)
    # length of the rows
    numberOfLines = len(fr.readlines())
    # form the matrix
    # zeros(2,3)-- get matrix of 2*3, all value equal to 0
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # get new str
        line = line.strip()
        # split by \t
        listFromLine = line.split('\t')
        # matrix for attributes/features by columns
        returnMat[index, :] = listFromLine[0:3]
        # lables for each instance
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector   
    
  # data normalization
  # use min-max normalization
  # y=(x-MinValue)/(MaxValue-MinValue)　　
  def autoNorm(dataSet):
    # minimum values, maximum values, range
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    norm_dataset = (dataset - minvalue) / ranges
    return normDataSet, ranges, minVals
 
 # test by calculating the error rate
 def datingClassTest():
    # ratio of the testing set -- hoRatio
    # ratio of the training set -- 1-hoRatio
    hoRatio = 0.1  
    # load data setfrom file
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  
    # data normalization
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m-- length of the rows
    m = normMat.shape[0]
    # numTestVecs-- m smaple size
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
