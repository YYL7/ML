# import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load the dataset
dataMat = []
fr = open("9. K-Means/testSet.txt") 
for line in fr.readlines():
    curLine = line.strip().split('\t')
    fltLine = list(map(float,curLine))   
    dataMat.append(fltLine)

# KMeans
km = KMeans(n_clusters=4) 
km.fit(dataMat) # fit
km_pred = km.predict(dataMat) # predict
centers = km.cluster_centers_ # center

# plot
plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=km_pred)
plt.scatter(centers[:, 1], centers[:, 0], c="r")
plt.show()
