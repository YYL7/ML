from numpy import *
import matplotlib.pyplot as plt

# load the dataset
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)

# PCA
def pca(dataMat, topNfeat=9999999):
    """pca
    Args:
        dataMat:      original data matrix
        topNfeat:     N features
    Returns:
        lowDDataMat:  data matrix after dimentionality reduction
        reconMat:     new matrix
    """

    # calculate mean value for each column
    meanVals = mean(dataMat, axis=0)
    # print('meanVals', meanVals)

    # minus mean value
    meanRemoved = dataMat - meanVals
    # print('meanRemoved=', meanRemoved)

    covMat = cov(meanRemoved, rowvar=0)

    # eigVals-value， eigVects-vector
    eigVals, eigVects = linalg.eig(mat(covMat))
    # print('eigVals=', eigVals)
    # print('eigVects=', eigVects)
    
    eigValInd = argsort(eigVals)
    # print('eigValInd1=', eigValInd)

    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # print('eigValInd2=', eigValInd)
    redEigVects = eigVects[:, eigValInd]
    # print('redEigVects=', redEigVects.T)
    # print( "---", shape(meanRemoved), shape(redEigVects))
    
    # new matrix
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print('lowDDataMat=', lowDDataMat)
    # print('reconMat=', reconMat)
    return lowDDataMat, reconMat

# replace all NaN with mean
def replaceNanWithMean():
    datMat = loadDataSet('11. PCA/secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # mean based on the data without NaN
        # .A, Returns an array based on the matrix
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # replace all NaN with mean
        datMat[nonzero(isnan(datMat[:, i].A))[0],i] = meanVal
    return datMat

# show picture
def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

# analyse data
def analyse_data(dataMat):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat-meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigvals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigvals)

    topNfeat = 20
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print('Principal Component：%s, Variance：%s%%, Cumulative Variance：%s%%' % (format(i+1, '2.0f'), format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))

Principal Component
if __name__ == "__main__":
    # # load the dataset，and coverte to float
    # dataMat = loadDataSet('11. PCA/testSet.txt')
    # # only one feature
    # lowDmat, reconMat = pca(dataMat, 1)
    # # two features
    # # lowDmat, reconMat = pca(dataMat, 2)
    # # print(shape(lowDmat))
    # show_picture(dataMat, reconMat)

    # use PCA to do dimentionality reduction
    dataMat = replaceNanWithMean()
    print(shape(dataMat))
    analyse_data(dataMat)
    # lowDmat, reconMat = pca(dataMat, 20)
    # print(shape(lowDmat))
    # show_picture(dataMat, reconMat)
