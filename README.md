# Personalized-Shopping-Recommender

## 1. Executive Summary

In this project, a cloud-based SaaS application was developed and deployed to provide target users guidance on online-shopping platforms. Our main objective is to free users from exhausting search among massive irrelevant products and instead provide them with tailored recommendations based on their own tastes yet with space for them to explore on their freewill. With a specific demonstrative use case, our service provides an interactive and lean interface, delivering highly reliable personalized recommended video gaming related products to users, enabling users to give feedback on the product ratings with registered accounts, and allowing users to search for specific products with specified price range and even with uploaded photos with a high matching accuracy innovatively. Our SaaS service is capable of on-demand self-service through the web interface, fulfilling rapid elastic demand between sales festivals and normal periods, handling failures, and securing the user's personal information and privacy. 

## 2. Recommendation System
### 2.1 Algorithms

The main objective of the SaaS service is to provide the users accurate recommendations of video games based on their past ratings. The PSR service is able to provide personalized recommendations for either existing users or new users who have no previous ratings with good quality by using collaborative filtering (CF), and most popular approaches like low-dimensional factor models:

● Memory-Based Collaborative Filtering: 
k-NN inspired algorithms: Many k-NN algorithms have been compared and contrasted before implementation, like k-NN baseline, k-NN with means, and k-NN with Z-score, etc. For each algorithm, the actual number of neighbors that are aggregated to compute an estimation is necessarily less than or equal to 𝑘. The predicted rating for user 𝑢 and game 𝑖, 𝑟̂𝑢𝑖 is calculated differently;

As no training or optimization is involved, it is an easy to use approach. But its performance decreases when we have sparse data which hinders scalability of this approach for most of the real-world problems. 

● Model based approach: 
CF models are developed using machine learning algorithms to predict user’s rating of unrated items. Matrix Factorization (MF): The idea behind such models is that attitudes or preferences of a user can be determined by a small number of hidden factors:

Matrix decomposition can be reformulated as an optimization problem with loss functions and constraints. Now the constraints are chosen based on the property of the model. Commonly used MF algorithms are SVD (Singular value decomposition), and NMF (Non negative matrix decomposition where non-negative elements are used in resultant matrices) were implemented in this recommender. Slope-One: Simple yet accurate algorithm. In this algorithm, the set of relevant items is created 𝑅𝑖(𝑢), i.e. the set of items 𝑗 rated by 𝑢 that also have at least one common user with 𝑖. And dev(𝑖, 𝑗) is defined as the average difference between the ratings of 𝑖 and those of 𝑗. The formula of predicting the rating is:
  
Co-clustering: In this algorithm, users and items are assigned some clusters 𝐶𝑢, 𝐶𝑖, and some co-clusters 𝐶𝑢𝑖.

### 2.2. Solution Methods & Implementation:

To enable the recommender system and make it capable of predicting the user rating accurately, a large dataset of video game ratings containing 1.3M rating samples was used to train and validate the model performance. The data contains user Id, video game Id, and the rating this user gives to the corresponding video game. Some explorative data analysis (EDA) has been performed on the dataset: The rating for the video games ranges from 1 to 5 as integers, and over half of the games have been rated as 5. In terms of rating per user, most of users only rated 1 or 2 games, which would make the user-item matrix very sparse. To reduce sparsity, only users who have rated for a certain number of items are kept. The model was then trained and validated on these extracted rating records (160K data samples). 
The algorithms mentioned in the last section were implemented using the surprise package, which is a Python scikit building and analyzing recommender systems that deal with explicit rating data. The models were trained and validated on the dataset using cross validation techniques (setting number of folds to 3, taking the two thirds of the data for each fold to train the model and validate the model performance on the left data). 

The result shows that SVDpp Algorithm produced the most accurate prediction of the ratings, the final validation RMSE is 1.06. This algorithm was then further trained on the dataset with hyper parameter tuning (learning rate, and number of epochs) with 5 folds cross validation. This further reduced the RMSE of prediction to 1.046. Last but not least, the model was tested on a completely untouched test data and the test RMSE was 1.042. This model was finally chosen for predicting the ratings (float between 1 and 5) of any existing video game from any existing user. On the other hand, k-NN model is more interpretable of generating the most similar users of a target user. Thus, the k-NN family algorithms were also fine-tuned to further reduce the RMSE. k-NN with means algorithm was finally chosen due to its best accuracy to get the top K users that are most similar to the target user based on their previous ratings.

The function of getting the Top N recommended video games of a specific user was then defined step by step: first get the top K most similar users of the target user; then use the SVDpp model to predict the ratings these K users would give for all the unique video games, and take the average of all K users as the final prediction rating; thereafter, rank the ratings to fetch the Top N items based on the prediction and output the results.


