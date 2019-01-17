# Singular Value Decomposition (SVD）
SVD, use to represent the original data set with a much smaller data set by removing noise and redundant information and thus to save bits (SVD, extracting the relevant features from a collection of noisy data.)

• Latent semantic indexing/ Latent semantic Analysis (LSI/LSA)

In LSI, a matrix is constructed of documents and words. When the SVD is done on this matrix, it creates a set of singular values, which represent concepts or topics contained in the documents. A simple search that looks only for the existence of words may have problems if the words are misspelled. Another problem with a simple search is that synonyms may be used and looking for the existence of a word wouldn’t tell you if a synonym was used to construct the document. If a concept is derived from thousands of similar documents, both synonyms will map to the same concept.

• Recommendation systems

Simple versions of recommendation systems compute similarity between items or people. More advanced methods use the SVD to create a theme space from the data and then compute similarities in the theme space.


Recommendation systems--Collaborative filtering:

Collaborative filtering works by taking a data set of users’ data and comparing it to the data of other users.

The only real math going on behind the scenes is the Similarity Measurement:

-- The Euclidian distances, examine the root of square differences between coordinates of pair of objects.

-- Pearson correlation, corrcoef(), normalize the range from 0 to 1.0 

-- Cosine Similarity measures the cosine of the angle between two vectors. 

Evaluating recommendation engines:

-- Cross-Validation. To take some known rating and hold it out of the data and then make a prediction for that value. We can compare our predicted value with the real value from the user. (Training and Testing Data)

-- Root Mean Squared Error (RMSE), computes the mean of the squared error and then takes the square root of that. If you’re rating things on a scale of one to five stars and you have an RMSE of 1.0, it means that your predictions are on average one star off of what people really think.

# Summary 

The singular value decomposition (SVD) is a powerful tool for dimensionality reduction. You can use the SVD to approximate a matrix and get out the important features. By taking only the top 80% or 90% of the energy in the matrix, you get the important features and throw out the noise. The SVD is employed in several applications today. One successful application is in recommendation engines. 

Recommendations engines recommend an item to a user. Collaborative filtering is one way of creating recommendations based on data of users’ preferences or actions. At the heart of collaborative filtering is a similarity metric. Several similarity metrics can be used to calculate the similarity between items or users. The SVD can be used to improve recommendation engines by calculating similarities in a reduced number of dimensions. 


