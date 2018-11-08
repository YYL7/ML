KNN


KNN method is mainly for classification. 
Given a training dataset with labels, KNN will help to predict the classification of the new data by calculating the distance with each pieces of the existing data and finding its k nearest neighbors. 
The majority is the new class we assign to the new data.


For every point in our dataset:
1. Calculate the distance between inX and current point
2. Sort the distances in increasing order
3. Take k items with lowest distance to inX
4. Find the majority class among these k items
5. Return the majority class as our predication for the class of inX


Pros:
High accuracy, Insensiitive to outliers, No assumptions about data
Cons:
Computationally expensive, require a lot of memory
