# ⭐Machine Learning
# ⭐Classification
Classification is a method of supervised learning  of identifying to which of a set of categories a new observation belongs, on the basis of a dataset with lables.
# 1. KNN
KNN method is mainly for classification. Given a training dataset with labels, KNN will help to predict the classification of the new data by calculating the distance with each pieces of the existing data and finding its k nearest neighbors. The majority is the new class we assign to the new data.
# 2. Decision Tree
Decision tree uses a tree-like model to describe the classification of the instances. A decision tree is drawn upside down with its root at the top. The decision blocks(rectangles) represents a condition/internal node, based on which the tree splits into branches/ edges. The end of the branch that doesn’t split anymore is the leaf, which represent various classes.  One of the best things about decision trees is that we can easily understand the data. 
# 3. Naive Bayes
KNN and Decision Tree give us exactly the classification, and Naïve Bayes will give us a guess of the classification about the class and assign a probability estimate to that best guess by Bayesian rules.
# 4. Logistic Regression
Logistic regression is trying to build an equation to do classification with the data we. For the logistic regression classifier (equation), we’ll take our features and multiply each one by a weight and then add them up. This result will be put into the sigmoid function, and we’ll get a number between 0 and 1. And we will use optimiza6. AdaBoosttion algorithms to find these best-fit parameters.
# 5. SVM(Support Vector Machine)
For SVM, we have separating hyperplane, which is used to separate the dataset. The points closest to the separating hyperplane are known as support vectors. So, SVM is to maximize the distance from the separating line to the support vectors by solving a quadratic optimization problem.

# ⭐Emsemble
Ensemble methods are meta algorithms that combine several machine learning techniques into one predictive model in order to decrease variance (bagging), bias (boosting), or improve predictions (stacking). Ensemble methods usually produces more accurate solutions than a single model would. 

Two types: Bagging (Random Forest) and Boosting (AdaBoost).

--Bagging (Building classifiers from randomly resampled data), the data is taken from the original dataset S times to make S new datasets. The datasets are the same size as the original. Each dataset is built by randomly selecting an example from the original with replacement. (same weight) (e.g. Random Forest)

--Boosting is different from bagging because the output is calculated from a weighted sum of all classifiers. The weights aren’t equal as in bagging but are based on how successful the classifier was in the previous iteration. (different weights so average weighted) (e.g. AdaBoost)

# 6. Random Forest
Random Forest is an Ensemble method. Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.

# 7. AdaBoost
AdaBoost uses a weak learner as the base classifier with the input data weighted by a weight vector. In the first iteration the data is equally weighted. But in subsequent iterations the data is weighted more strongly if it was incorrectly classified previously. This adapting to the errors is the strength of AdaBoost.

# ⭐Regression

# 8. Regression
Regression is the process of predicting a target value for continuous data. discrete in classification. Minimizing the sum-of-squares error is used to find the best weights for the input features in a regression equation. 

# ⭐Clustering
Clustering is a type of unsupervised learning that automatically forms clusters of similar things.

# 9. K-Means
K-Means is an algorithm that will find k clusters for a given dataset without labels. The number of clusters k is user defined. Each cluster is described by a single point known as the centroid. Centroid means it’s at the center of all the points in the cluster.

# ⭐Association rule
Association analysis is the task of finding relationships in large datasets. These relationships can take two forms: frequent item sets or association rules. Frequent item sets are a collection of items that frequently occur together. The second one is Association rules suggest that a strong relationship exists between two items.

# 10. Aprori
Apriori is an algorithm for frequent item set mining and association rule learning over transactional databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets if those item sets appear sufficiently often in the database.

# ⭐Dimensionality Reduction 
Reduce the data from higher dimensions to lower dimensions, which will be easier for us to deal with data. (Preprocessing data)

# 11. Principal Component Analysis (PCA)
PCA uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables.

# 12. Singular Value Decomposition (SVD）
SVD, use to represent the original data set with a much smaller data set by removing noise and redundant information and thus to save bits (SVD, extracting the relevant features from a collection of noisy data.)

# ⭐Big data and MapReduce
When your computing needs have exceeded the capabilities of your computing resources, one solution to this is to break up your computing into parallel jobs such as MapReduce, by which you could break your jobs into map and reduce steps.

# 13. MapReduce
MapReduce, a software framework for spreading a single computing job across multiple computers to shorten the time.


