import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm



np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20

# load the dataset
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


X, Y = loadDataSet('5. SVM -Support Vector Machine/testSet.txt')
X = np.mat(X)

print("X=", X)
print("Y=", Y)

# SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# seperating hyperplane
w = clf.coef_[0]

# Slope
a = -w[0] / w[1]

# from -2 to 10, sampling 50 samples in sequence
xx = np.linspace(-2, 10)  # num=50)

# Two-dimensional linear equation
yy = a * xx - (clf.intercept_[0]) / w[1]
print("yy=", yy)

# plot the parallels to the separating hyperplane that pass through the support vectors
print("support_vectors_=", clf.support_vectors_)
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
plt.scatter(X[:, 0].flat, X[:, 1].flat, c=Y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()
