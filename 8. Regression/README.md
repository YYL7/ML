# Regression

# Linear Regression

Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables.  The Linear regression equation is to add up the inputs which multiplied by weights to get the output.

Input data is in the matrix X, and the regression weights in the vector w. For a given piece of data X1 our predicted value is given by y1 = X1^T w. We have the Xs and ys, but how can we find the ws? One way is to find the ws that minimize the error. We define error as the difference between predicted y and the actual y. Using just the error will allow positive and negative values to cancel out, so we use the squared error.

# Locally Weighted Linear Regression (LWLR)

One problem with linear regression is that it tends to underfit the data by giving the lowest mean-squared error for unbiased estimators. One way to reduce the mean-squared error is a technique known as locally weighted linear regression (LWLR). In LWLR we give a weight to data points near our data point of interest; then we compute a least-squares regression.

One problem with locally weighted linear regression is that it involves numerous computations. You have to use the entire dataset to make one estimate.

# If we have more features than data points,
when we compute (X^TX)-1, we’ll get an error because it is not full rank.

Three ways: Ridge Regression, The lasso, Forward stagewise regression

Add bias into our estimations, giving us a better estimate. (Decrease the variance of the models)

# Ridge Regression

Ridge regression adds an additional matrix λI to the matrix XTX so that it’s non-singular, and we can take the inverse of the whole thing: XTX + λI.

The matrix I is a maximum identity matrix where there are 1s in the diagonal elements and 0s elsewhere. 

The symbol λ is a user-defined scalar value. We can use the λ value to impose a maximum value on the sum of all our ws. By imposing this penalty, we can decrease unimportant parameters. (This decreasing is known as shrinkage in statistics.) 

The sum of the squares of all our weights must be less than or equal to λ.

# The lasso

The sum of the absolute value of all our weights must be less than or equal to λ.  Taking the absolute value instead of the square of all the weights.

If λ is small enough, some of the weights are forced to be exactly 0, which makes it easier to understand our data.

# Forward stagewise regression
A greedy algorithm in that at each step it makes the decision that will reduce the error the most at that step. 

Initially, all the weights are set to 0. The decision that’s made at each step is increasing or decreasing a weight by some small amount.

# Summary 

Regression is the process of predicting a target value for continuous data. discrete in classification. Minimizing the sum-of-squares error is used to find the best weights for the input features in a regression equation. 
