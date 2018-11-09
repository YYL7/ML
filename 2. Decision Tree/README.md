Decision Tree

Decision tree uses a tree-like model to describe the classification of the instances. 
A decision tree is drawn upside down with its root at the top. The decision blocks(rectangles) 
represents a condition/internal node, based on which the tree splits into branches/ edges. 
The end of the branch that doesn’t split anymore is the leaf, which represent various classes.  
One of the best things about decision trees is that we can easily understand the data. 


Check if every item in the dataset is in the same class in Decision Tree:
1. If so return the class label
2. Else
find the best feature to split the data;
split the dataset;
create a branch node;
for each split:
call createBranch and add the result to the branch node;
return branch node



Information theory:
A measure of information, representing the degree of uncertainty.


Information Gain:
The change of information before and after the split.
  The split with highest information gain is the best option.


Entropy:
The expected value of all the information of all possible values of our class.
Quantify the amount of uncertainty involved in the value of a random variable.  
The split with lowest entropy is the best option.


Pros:
1. Computationally cheap to use
2. Easy for humans to understand learned results
3. Missing values OK
4. Can deal with irrelevant features

Cons:
Prone to overfitting
