from numpy import *
import matplotlib.pylab as plt
from time import sleep
import bs4
from bs4 import BeautifulSoup
import json
import urllib.request   

# load the dataset
def loadDataSet(fileName):
    Returns：
        dataMat ：  data with feature columns
        labelMat ： data with lable column
    """
    # number of the features, exclude the lable column
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# Linear regression 
def standRegres(xArr, yArr):
    '''
    Args:
        xArr ：data with feature columns
        yArr ：data with lable column
    Returns:
        ws：weights
    '''

    # converting xArr and yArr to matrix by mat(), then transpose the y matrix
    xMat = mat(xArr)
    yMat = mat(yArr).T 
    xTx = xMat.T * xMat

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # weights
    ws = xTx.I * (xMat.T * yMat)
    return ws

# Locally Weighted Linear Regression (LWLR)
# In LWLR we give a weight to data points near our data point of interest; then we compute a least-squares regression. 
def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
        Args：
            testPoint：test point
            xArr： features
            yArr： lables
            k:     constant k that will determine how much to weight nearby points.
        Returns:
            testPoint * ws：  predictions

    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # Get the number of rows in the xMat matrix
    m = shape(xMat)[0]
    # eye(), return a two-dimensional array with a diagonal element of 1 and other elements of 0,
    #        and creating a weight matrix weights that initializes a weight for each sample point
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# test on LWLR
def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
        Args：
            testArr：testing data
            xArr：   feature columns
            yArr：   lable column
            k：      constant k that will determine how much to weight nearby points.
        Returns：
            yHat：   predictions
    '''
    # number of the testing data
    m = shape(testArr)[0]
    # creating a 1 * m  matric with all 0
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# ploting
def lwlrTestPlot(xArr, yArr, k=1.0):
    '''
        Args： 
            xArr：feature columns
            yArr：lable column
            k：constant k that will determine how much to weight nearby points.
        Return：
            yHat：predicted value
            xCopy：copy of the xArr
    '''
    yHat = zeros(shape(yArr))
    # coverting the xArr to matrix
    xCopy = mat(xArr)
    # sorting
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy

# residual sum of squares (RSS)
def rssError(yArr, yHatArr):
    '''
        Args:
            yArr：actual value (lable)
            yHatArr：predicted value
        Returns:
            residual sum of squares (RSS)
    '''
    return ((yArr - yHatArr) ** 2).sum()

# Ridge Regression
# Ridge regression adds an additional matrix λI to the matrix XTX so that it’s non-singular
def ridgeRegres(xMat, yMat, lam=0.2):
    '''
        Args：
            xMat：features columns
            yMat：lable columns
            lam：lambda
        Returns：
            weights after Ridge Regression
    '''

    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

# test on Ridge Regression
def ridgeTest(xArr, yArr):
    '''
        Args：
            xArr：feature columns
            yArr：lable column
        Returns：
            wMat：Output all regression coefficients to a matrix 
    '''

    xMat = mat(xArr)
    yMat = mat(yArr).T
    # yMean, mean of y
    yMean = mean(yMat, 0)
    # All features  of Y minus the mean
    yMat = yMat - yMean

    xMeans = mean(xMat, 0)
    # xVar, variance
    xVar = var(xMat, 0)
    # All features are subtracted from their respective mean values and divided by the variance
    xMat = (xMat - xMeans) / xVar

    numTestPts = 30
    # creating matrix of 30 * m with 0
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        # exp(), return e^x
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

# regularize
def regularize(xMat):  
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # mean
    inVar = var(inMat, 0)  # variance
    inMat = (inMat - inMeans) / inVar
    return inMat

# Forward stagewise regression
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


