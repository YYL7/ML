from numpy import *
import matplotlib.pyplot as plt


class optStruct:
    """
    Establish a data structure to hold all important values
    """
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
        Args:
            dataMatIn    :data matrix
            classLabels  :lables
            C       :Slack variables (constant values) 
	            allow some data points to be on the wrong side of the plane
            toler   :rate of tolerance
            kTup    :tuple in the kernel function
        """

        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler

        # m: length of the rows
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0

        # Error cache, the first column shows whether eCache is valid, and the second column gives the actual E value.
        self.eCache = mat(zeros((self.m, 2)))

        # Matrix m*m
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)

# Kernel 
# calc the kernel or transform data to a higher dimensional space
def kernelTrans(X, A, kTup):  
    """
    Args:
        X     dataMatIn
        A     the data of ith in the dataMatIn
        kTup  kenerl fuction
    Returns:
    """
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        # linear kernel:   m*n * n*1 = m*1
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # Guassian
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

# load the dataset
def loadDataSet(fileName):
    """loadDataSet
    Args:
        fileName 
    Returns:
        dataMat  
        labelMat
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# to calculate error 
def calcEk(oS, k):
    """calcEk（E=preditted value - actual value）
    Args:
        oS  :optStruct
        k   :specific row
    Returns:
        Ek  :error (preditted value - actual value)
    """
    fXk = multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# randomly select a integer
def selectJrand(i, m):
    """
    Args:
        i  
        m  :number of alpha
    Returns:
        j  :retern a random number from the range of (0,m), except i
    """
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j

# select j and return the optimal j and Ej
def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    """
    Args:
        i   :randonly select the row of ith 
        oS  :optStruct
        Ei  :Error Ei (Predicted - Actual)
    Returns:
        j   :randonly select the row of jth 
        Ej  :Error Ej (Predicted - Actual)
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # Loop over all the values and select the one that makes the change the most
            if k == i:
                continue  # don't calc for i, waste of time

            # Error Ek (Predicted - Actual)
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                # Choose j with the largest step size
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # If it is the first loop, randomly select an alpha value
        j = selectJrand(i, oS.m)

        # Error Ej (Predicted - Actual)
        Ej = calcEk(oS, j)
    return j, Ej

# update Error Ek
def updateEk(oS, k):
    """
    Args:
        oS  :optStruct
        k   :The line number of a column
    """

    # Error Ek (Predicted - Actual)
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

# clip Alpha (L<=aj<=H)
def clipAlpha(aj, H, L):
    """
    Args:
        aj  :Target value
        H   :Highest
        L   :Lowest
    Returns:
        aj  :Target value
    """
    aj = min(aj, H)
    aj = max(L, aj)
    return aj

# Inner loop code
def innerL(i, oS):
    """innerL
    Args:
        i   :ith row
        oS  :optStruct
    Returns:
        0   :fail to get the Optimal value 
        1   :get the Optimal value ，and save to oS.Cache
    """

    # Error Ei (Predicted - Actual)
    Ei = calcEk(oS, i)
    '''
    yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    yi*f(i) == 1 and 0<alpha< C (on the boundary)
    yi*f(i) <= 1 and alpha = C (between the boundary)
    '''
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # Select the j corresponding to the largest error for optimization.
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # clip alphas[j] to the range of 0-C. If L==H, do nothing, and return 0
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0

        # eta is the Optimal modification of alphas[j], if eta==0，exit the for loop
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0

        # new alphas[j]
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # adjust
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # update
        updateEk(oS, j)

        # Check if alpha[j] is only a slight change, and if so, exit the for loop.
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            # print("j not moving enough")
            return 0

        # alphas[i] and alphas[j] are changed as well, although the size of the change is the same, but the direction of change is just the opposite.
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # update
        updateEk(oS, i)

        # assign a constant b
        # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
        # so：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0

# SMO (Sequential Minimal Optimization)
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    Args:
        dataMatIn    :features
        classLabels  :lables
        C   :Slack variables (constant values) 
	     allow some data points to be on the wrong side of the plane.
        toler   :rate of tolerance
        maxIter :maximum iteration
        kTup    :tuple in the kernel function
    Returns:
        b       :Constant value of the model
        alphas  :Lagrange multiplier
    """

    # optStruct 
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                # if alpha exist, then +1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
	
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

# calculate weights
def calcWs(alphas, dataArr, classLabels):
    """
    Args:
        alphas        :Lagrange multiplier
        dataArr       :features
        classLabels   :lables
    Returns:
        wc  :weights
    """
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i].T)
    return w

# Test
# RBF (Radial Bias Function) 
# a function that takes a vector and outputs a scalar based on the vector’s distance.
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('db/6.SVM/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]  # get matrix of only support vectors
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))

        # fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadDataSet('db/6.SVM/testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

# image to vector
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# load the image
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    print(dirName)
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

# RBF (Radial Bias Function) 
# a function that takes a vector and outputs a scalar based on the vector’s distance.
def testDigits(kTup=('rbf', 10)):
	
    # 1. training dataset
    dataArr, labelArr = loadImages('db/6.SVM/trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    # print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        # 1*m * m*1 = 1*1 
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    # 2. testing dataset
    dataArr, labelArr = loadImages('db/6.SVM/testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

# plot
def plotfig_SVM(xArr, yArr, ws, b, alphas):
    xMat = mat(xArr)
    yMat = mat(yArr)

    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])

    x = arange(-1.0, 10.0, 0.1)

    # x.w + b = 0, w0.x1 + w1.x2 + b = 0, x2=y
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')

    # find the SVM and make it red
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()
