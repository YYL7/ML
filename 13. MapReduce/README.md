# Big data and MapReduce

# Hadoop

A free, open source implementation of the MapReduce framework written in Java, for distributing data processing to multiple machines.

# MapReduce
A software framework for spreading a single computing job across multiple computers to shorten the time.


MapReduce is done on a cluster, which is made up of nodes. MapReduce works like this: 

•	The master node handles the whole MapReduce job.

•	The first step is the Map step. A single job is broken down into small sections, and the input data is chopped up and distributed to each node. (Each node operates on only its data.) The code that’s run on each node is called the mapper.

•	The output from the individual mappers is Sorted and then Combined in some way. Then data is then broken into smaller portions and distributed to the nodes for further processing.  (Data is passed in key/value pairs.)

•	The second processing step is the reduce step, and the code run is known as the reducer. The output of the reducer is the final answer you’re looking for.


# Machine learning in MapReduce

•	Naïve Bayes— Can be naturally implementable in MapReduce. In MapReduce, it’s easy to calculate sums. In naïve Bayes, we were calculating the probability of a feature given a class. We can give the results from a given class to an individual mapper. We can then use the reducer to sum up the results. 

•	k-Nearest Neighbors (KNN)—Trying to find similar vectors in a small dataset can take a large amount of time. In a massive dataset, it can limit daily business cycles. One approach to speed this up is to build a tree, such as a tree to narrow the search for closest vectors. This works well when the number of features is under 10. A popular method for performing a nearest neighbor search on higher-dimensional items such as text, images, and video is locality-sensitive hashing. 

•	Support vector machines (SVM)—The Platt SMO algorithm may be difficult to implement in a MapReduce framework. There are other implementations of SVMs that use a version of stochastic gradient descent such as the Pegasos algorithm. There’s also an approximate version of SVM called proximal SVM, which computes a solution much faster and is easily applied to a MapReduce framework.

•	Singular value decomposition (SVD)—The Lanczos algorithm is an efficient method for approximating eigenvalues. This algorithm can be applied in a series of MapReduce jobs to efficiently find the singular values in a large matrix. The Lanczos algorithm can similarly be used for principal component analysis. 

•	k-means clustering—One popular version of distributed clustering is known as canopy clustering. You can calculate the k-means clusters by using canopy clustering first and using the canopies as the k initial clusters.

# Summary

When your computing needs have exceeded the capabilities of your computing resources, one solution to this is to break up your computing into parallel jobs such as MapReduce, by which you could break your jobs into map and reduce steps.

Data is passed between the mapper and reducers with key/value pairs. Typically, data is sorted by the value of the keys after the map step. Hadoop is a popular Java project for running MapReduce jobs. Hadoop has an application for running non-Java jobs called Hadoop Streaming.

A number of machine learning algorithms can be easily written as MapReduce jobs. Some machine learning jobs need to be creatively redefined in order to use them in MapReduce. Support vector machines are a powerful tool for text classification but training a classifier on a large number of documents can involve a large amount of computing resources. One approach to creating a distributed classifier for support vector machines is the Pegasos algorithm. Machine learning algorithms that may require multiple MapReduce jobs such as Pegasos are easily implemented in mrjob.


