from numpy import linalg as la
from numpy import *

# load dataset
def loadExData3():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

# load dataset
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData():
    return[[0, -1.6, 0.6],
           [0, 1.2, 0.8],
           [0, 0, 0],
           [0, 0, 0]]


# calculate similarity, Eclidean distance
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


# corrcoef, normalize the range from 0 to 1.0 
def pearsSim(inA, inB):
    # If it does not exist, returns 1.0, the two vectors are fully correlated.
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# Calculating cosine similarity
# If 90 degrees, the similarity is 0; if two vectors have same direction, the similarity is 1.0.
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)


# recommandation engine--Based on item similarity
def standEst(dataMat, user, simMeas, item):
    """
    Args:
        dataMat:         training dataset
        user:            user 
        simMeas:         similarity
        item:            Unrated item 
    Returns:
        ratSimTotal/simTotal     score（0～5）
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        # if rating with 0, then skip
        if userRating == 0:
            continue
        # Find items that both users have rated
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # if similarity is 0，then no overlap
        if len(overLap) == 0:
            similarity = 0
        # if exist overlap, then recalculate the similarity based on the overlap
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is : %f'(iten,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # normalization (divide by simTotal)
    else:
        return ratSimTotal/simTotal


# SVD
def svdEst(dataMat, user, simMeas, item):
    """
    Args:
        dataMat:         training dataset
        user:            user 
        simMeas:         similarity
        item:            Unrated item 
    Returns:
        ratSimTotal/simTotal     score（0～5）
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)

    # analyse_data(Sigma, 20)

    Sig4 = mat(eye(4) * Sigma[: 4])

    # U matrix
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    print('dataMat', shape(dataMat))
    print('U[:, :4]', shape(U[:, :4]))
    print('Sig4.I', shape(Sig4.I))
    print('VT[:4, :]', shape(VT[:4, :]))
    print('xformedItems', shape(xformedItems))

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # similarity
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


# recommend(), given a user, it will return the top N best recommendations for that user.
# default value for N is 3
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """svdEst( )
    Args:
        dataMat:         training dataset
        user:            user
        simMeas:         similarity
        estMethod:       Recommended algorithm used
    Returns:
        return the top N best recommendations
    """
    # unrated Items
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    # If there is no unrated item, then exit the function
    if len(unratedItems) == 0:
        return 'you rated everything'
    # item Scores
    itemScores = []
    # for loop with unrated Items
    for item in unratedItems:
        # estimated Score  
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # Reverse sorting, get the top N unrated items for recommendation
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]

# analyse data
def analyse_data(Sigma, loopNum=20):
    Sig2 = Sigma**2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i+1])
        print('Principal component：%s, variance：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '4.2f')))


# Image data
def imgLoadData(filename):
    myl = []
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    return myMat


# print matrix
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,)
            else:
                print(0,)
        print('')


# Image compression
def imgCompress(numSV=3, thresh=0.8):
    myMat = imgLoadData('12. SVD/0_5.txt')

    print("****original matrix****")
    printMat(myMat, thresh)

    U, Sigma, VT = la.svd(myMat)
    # SigRecon = mat(zeros((numSV, numSV)))
    # for k in range(numSV):
    #     SigRecon[k, k] = Sigma[k]

    # analyse data
    analyse_data(Sigma, 20)

    SigRecon = mat(eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values *****" % numSV)
    printMat(reconMat, thresh)


if __name__ == "__main__":

    # Data = loadExData()
    # print('Data:', Data)
    # U, Sigma, VT = linalg.svd(Data)
    # print('U:', U)
    # print('Sigma', Sigma)
    # print('VT:', VT)
    # print('VT:', VT.T)

    # # 3x3
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])

    myMat = mat(loadExData3())
    # print(myMat)
    # The first way to calculate similarity
    print(recommend(myMat, 1, estMethod=svdEst))
    # The second way to calculate similarity
    print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

    # recommend
    print(recommend(myMat, 2))
