# AdaBoost (Adaptive Boosting)

AdaBoost uses a weak learner as the base classifier with the input data weighted by a weight vector. In the first iteration the data is equally weighted. But in subsequent iterations the data is weighted more strongly if it was incorrectly classified previously. This adapting to the errors is the strength of AdaBoost.

How it work:

A weight is applied to every example in the training data. We’ll call the weight vector D. Initially, these weights are all equal. A weak classifier is first trained on the training data. The errors from the weak classifier are calculated, and the weak classifier is trained a second time with the same dataset. This second time the weak classifier is trained, the weights of the training set are adjusted so the examples properly classified the first time are weighted less and the examples incorrectly classified in the first iteration are weighted more. To get one answer from all these weak classifiers, AdaBoost assigns  values to each of the classifiers. The  values are based on the error of each weak classifier.

Pros:

--Low generalizaton error

--Easy to code

--Works with most classifiers

--No parameters to adjust

Cons:

--Sensitive to outliers



# Classification Imbalance
Data that doesn’t have an equal number of positive and negative examples. The problem also exists when the costs for misclassification are different from positive and negative examples. 

Firstly, can we get more data?

Secondly, we could use other measures.

•	Precision and Recall as metrics to measure the performance classifiers when classification of one class is more important than classification of the other class. 

•	ROC curves to evaluate different classifiers (ROC=1 means best)
X-axis: FP/(FP+TN)
Y-axis: TP/(TP+FN)  (Sensitivity/Recall)

Thirdly, try resampling.

•	Undersampling for the large dataset

•	Oversampling for the small dataset
