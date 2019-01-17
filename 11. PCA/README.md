# Dimensionality Reduction 降维 
•	Reduce the data from higher dimensions to lower dimensions, which will be easier for us to deal with data. (Preprocessing data)

•	For example, converting the million pixels on the monitor into a three-dimensional image, showing the object’s position on the current environment in real time. (Reduce the data from one million dimensions to three)

•	We must identify the relevant features before we can begin to apply other machine learning algorithms.


Reasons for Dimensionality Reduction:

•	Displaying data and results

•	Making the dataset easier to use 

•	Reducing computational cost of many algorithms

•	Removing noise 

•	Making the results easier to understand



# Principal Component Analysis (PCA)
PCA uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables.) 

•	The dataset is transformed from its original coordinate system to a new coordinate system. The new coordinate system is chosen by the data itself. 

•	The first new axis is chosen in the direction of the most variance in the data. The largest variation is the data telling us what’s most important.

•	The second axis is orthogonal to the first axis and in the direction of an orthogonal axis with the largest variance. 

•	This procedure is repeated for as many features as we had in the original data. 

•	Find that the majority of the variance is contained in the first few axes. Therefore, we can ignore the rest of the axes, and we reduce the dimensionality of our data.

# Summary

•	PCA allows the data to identify the important features. It does this by rotating the axes to align with the largest variance in the data. Other axes are chosen orthogonal to the first axis in the direction of largest variance. Eigenvalue analysis on the covariance matrix can be used to give us a set of orthogonal axes. 

•	The PCA algorithm loads the entire dataset into memory. 



