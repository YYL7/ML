Logistic Regression
Logistic regression is trying to build an equation to do classification with the data we. 
For the logistic regression classifier (equation), we’ll take our features and multiply 
each one by a weight and then add them up. This result will be put into the sigmoid function, 
and we’ll get a number between 0 and 1. And we will use optimization algorithms to find these
best-fit parameters.


Gradient ascent 
If we want to find the maximum point on a function, then the best way to move is in the direction of the gradient.


Summary 

--Logistic regression is finding best-fit parameters to a nonlinear function called the sigmoid.
Methods of optimization can be used to find the best-fit parameters. Among the optimization algorithms, 
one of the most common algorithms is gradient ascent, which can be simplified with stochastic gradient ascent. 

--Stochastic gradient ascent can do as well as gradient ascent using far fewer computing resources.
In addition, stochastic gradient ascent can update what it has learned as new data comes in 
rather than reloading all the data as in batch processing.

