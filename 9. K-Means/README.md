# Clustering
Clustering is a type of unsupervised learning that automatically forms clusters of similar things.

# K-Means
k-means is an algorithm that will find k clusters for a given dataset without labels. The number of clusters k is user defined. Each cluster is described by a single point known as the centroid. Centroid means it’s at the center of all the points in the cluster.

First, the k centroids are randomly assigned to a point. 

Next, each point in the dataset is assigned to a cluster. The assignment is done by finding the closest centroid and assigning the point to that cluster. 

# How do you know that the clusters are good clusters?

The reason that k-means converged but we had poor clustering was that k-means converges on a local minimum, not a global minimum. 

We can use SSE (Sum of Squared Error, the sum of the values in column 1 of clusterAssment) to measure the quality of the cluster assignments.  Because the error is squared, this places more emphasis on points far from the centroid. A lower SSE means that points are closer to their centroids. One sure way to reduce the SSE is to increase the number of clusters. This defeats the purpose of clustering, so assume that you must increase the quality of your clusters while keeping the number of clusters constant.

# Bisecting k-means
To avoid the problem of local minimum, Bisecting k-means starts out with one cluster and then splits the cluster in two. It then chooses a cluster to split, which is decided by minimizing the SSE. This splitting based on the SSE is repeated until the user-defined number of clusters is attained.

# Summary
Clustering is a technique used in unsupervised learning. With unsupervised learning you don’t know what you’re looking for, no target variables. Clustering groups data points together, with similar data points in one cluster and dissimilar points in a different group. 

One widely used clustering algorithm is k-means, where k is a user-specified number of clusters to create. The k-means clustering algorithm starts with k-random cluster centers known as centroids. Next, the algorithm computes the distance from every point to the cluster centers. Each point is assigned to the closest cluster center. The cluster centers are then recalculated based on the new points in the cluster. This process is repeated until the cluster centers no longer move. This simple algorithm is quite effective but is sensitive to the initial cluster placement. To provide better clustering, a second algorithm called bisecting k-means can be used. 

Bisecting k-means starts with all the points in one cluster and then splits the clusters using k-means with a k of 2. In the next iteration, the cluster with the largest error is chosen to be split. This process is repeated until k clusters have been created. Bisecting k-means creates better clusters than k-means. 
