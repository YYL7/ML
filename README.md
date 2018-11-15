# Machine Learning
# 1.KNN
KNN method is mainly for classification. Given a training dataset with labels, KNN will help to predict the classification of the new data by calculating the distance with each pieces of the existing data and finding its k nearest neighbors. The majority is the new class we assign to the new data.
# 2. Decision Tree
Decision tree uses a tree-like model to describe the classification of the instances. A decision tree is drawn upside down with its root at the top. The decision blocks(rectangles) represents a condition/internal node, based on which the tree splits into branches/ edges. The end of the branch that doesn’t split anymore is the leaf, which represent various classes.  One of the best things about decision trees is that we can easily understand the data. 
# 3. Naive Bayes
KNN and Decision Tree give us exactly the classification, and Naïve Bayes will give us a guess of the classification about the class and assign a probability estimate to that best guess by Bayesian rules.
# 4. Logistic Regression
Logistic regression is trying to build an equation to do classification with the data we. For the logistic regression classifier (equation), we’ll take our features and multiply each one by a weight and then add them up. This result will be put into the sigmoid function, and we’ll get a number between 0 and 1. And we will use optimization algorithms to find these best-fit parameters.
# 5. SVM(Support Vector Machine)
For SVM, we have separating hyperplane, which is used to separate the dataset. The points closest to the separating hyperplane are known as support vectors. So, SVM is to maximize the distance from the separating line to the support vectors by solving a quadratic optimization problem.
