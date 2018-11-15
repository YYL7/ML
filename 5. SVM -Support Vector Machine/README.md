SVM (Support Vector Machine)

For SVM, we have separating hyperplane, which is used to separate the dataset. 
The points closest to the separating hyperplane are known as support vectors. 
So, SVM is to maximize the distance from the separating line to the support vectors
by solving a quadratic optimization problem.


SMO （Sequential Minimal Optimization)

--The SMO algorithm works to find a set of alphas and b. Once we have a set of alphas,
we can easily compute our weights w and get the separating hyperplane. 
--Here’s how the SMO algorithm works: it chooses two alphas to optimize on each cycle. 
Once a suitable pair of alphas is found, one is increased and one is decreased. 
To be suitable, a set of alphas must meet certain criteria. One criterion a pair 
must meet is that both alphas must be outside their margin boundary. 
The second criterion is that the alphas aren’t already clamped or bounded.

Kernel

--Using kernels for more complex data, to transform our data into a form that’s easily understood by our classifier. 
--The nonlinear problem in low-dimensionality becomes a linear problem in high-dimensionality. 

----mapping from one feature space to another feature space.

-- One great thing about the SVM optimization is that all operations can be written in terms of inner products.
Inner products are two vectors multiplied together to yield a scalar or single number. 
We can replace the inner products with our kernel functions without making simplifications. 
Replacing the inner product with a kernel is known as the kernel trick or kernel substation.

-- RBF (Radial Bias Function) (a function that takes a vector and outputs a scalar based on the vector’s distance.)


Summary

--Support vector machines are a type of classifier. SVM try to maximize margin by solving a 
quadratic optimization problem. SMO algorithm, which allowed fast training of SVMs by 
optimizing only two alphas at one time. (Full Platt version and the simplified version.)

--Support vector machines are a binary classifier and additional methods can be extended
to classification of classes greater than two. The performance of an SVM is also 
sensitive to optimization parameters and parameters of the kernel used.

--Support vectors have good generalization error: they do a good job of learning and 
generalizing on what they’ve learned. These benefits have made support vector machines popular, 
and they’re considered by some to be the best stock algorithm in unsupervised learning.
