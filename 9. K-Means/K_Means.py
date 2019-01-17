from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt

# load the dataset
def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # fltLine = [float(x) for x in curLine]
        fltLine = list(map(float,curLine))   
        dataSet.append(fltLine)
    return dataSet      # return dataMat

# Euclidean distance 
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# random centroid
def randCent(dataMat, k):
    # m-number of rows; n-number of features
    m, n = shape(dataMat)
    # create matrix with k*n (all 0)
    centroids = mat(zeros((k, n)))
    for j in range(n):
        # calculate the min value for each column(features)
        minJ = min(dataMat[:, j])
        # calculate the range
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # calculate the centroids
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

# K-Means
# Create K centroids, then assign each point to the nearest centroid and recalculate the centroid. 
# repeat the process until the cluster assignment result no longer changes.
def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    m, n = shape(dataMat)
    # clusterAssment, first column-cluster assignment results, second column-error(distance btw the current point to the centroid 
    clusterAssment = mat(zeros((m, 2)))  
    # create random K centroid 
    centroids = createCent(dataMat, k)

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # iterate all the data points to find the nearest centroid 
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # Calculate the distance btw the data point and the centroid
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # update the minDist if distJI is less than minDist
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # Update clusterChanged if the cluster assignment result changes 
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # update clusterAssment 
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # iterate all the data points and update centroids
        for cent in range(k):
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # calculate the mean
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# Bisecting k-means
# Bisecting k-means starts out with one cluster and then splits the cluster in two. 
# It then chooses a cluster to split, which is decided by minimizing the SSE. 
# This splitting based on the SSE is repeated until the user-defined number of clusters is attained.
def biKmeans(dataMat, k, distMeas=distEclud):
    m, n = shape(dataMat)
    # # clusterAssment, first column-cluster assignment results, second column- squared error
    clusterAssment = mat(zeros((m, 2)))
    # Calculate the centroid of the entire data set and save to a list 
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    centList = [centroid0]
    # iterate all points to calculate the error value of each point to the centroid
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataMat[j, :]) ** 2
    # repeate until the user-defined number of clusters is attained.
    while (len(centList) < k):
        # initialize the lowest SSE to inf 
        lowestSSE = inf
        # chooses a cluster to split, which is decided by minimizing the SSE
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
            # k-means starts out with one cluster and then splits the cluster in two. and also give the error
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # sum of the error will be the error for the split
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            # if we have SSE less than lowestSSE, then keep the split
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
 
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # update the best centroid to split
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # update centList, list of centroids
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

# Distance by Spherical law of cosines
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clusterClubs(fileName, imgName, numClust=5):
    '''
    fileName: path for the file
    imgName: path for the image 
    numClust: number of clusters
    :return:
    '''
    datList = []
    # the 4th column (latitude) and the 5th column (longitude) of the flie
    for line in open(fileName).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    # use biKmeans ablve and use use distSLC for calculating the distance
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]

    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread(imgName)
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
