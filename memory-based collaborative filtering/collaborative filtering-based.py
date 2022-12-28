"""Item-Based
We find similar movies based on other users' preferences."""
"""Contrarily to user-based methods, item similarity matrices tend to be smaller, 
which will reduce the cost of finding neighbours in our similarity matrix.

Also, since a single item is enough to recommend other similar items, 
this method will not suffer from the cold-start problem.
A drawback of item-based methods is that they there tends to be a lower 
diversity in the recommendations as opposed to user-based CF."""
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from pyspark import SparkContext, SQLContext

# from pyspark.sql.functions import *
# from pyspark.sql import functions as F

# from pyspark.ml.recommendation import ALS
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Importing Music Ratings Dataset
ratings_all = pd.read_csv('data/ratings.csv')

user_ratings_count = ratings_all.groupby(['user_id']).count()['song_id']
song_ratings_count = ratings_all.groupby(['song_id']).count()['user_id']

print(user_ratings_count)
print(song_ratings_count)

f, (ax1, ax2) = plt.subplots(1,2)

sns.histplot(user_ratings_count,stat='count', ax = ax1)
ax1.set_title('Frequency Distribution of Ratings by User')
sns.histplot(song_ratings_count, ax = ax2)
ax2.set_title('Frequency Distribution of Ratings per Song')

ax1.set_xlabel('Number of Ratings')
ax2.set_xlabel('Number of Ratings')
plt.subplots_adjust(0.1, 0.1, 2, 1.4)
plt.show()

print('Ratings Provided by Users:')
print('Average of Ratings Count =', np.round(np.mean(user_ratings_count), 4))
print('Median of Ratings Count =', np.round(np.median(user_ratings_count), 4))

print('\nRatings Received by Songs:')
print('Average of Ratings Count =', np.round(np.mean(song_ratings_count), 4))
print('Median of Ratings Count =', np.round(np.median(song_ratings_count), 4))

def data_sampling (df, item_nos=500, item_split=[0.90,0.10]):
    
    # Data preprocessing from user perspective
    
    # Frequency of song rating by each user
    user_rtgs_cnt = (df.groupby(['user_id']).count()).iloc[:,0:1].reset_index().rename(columns={"song_id":"rating_cnt"})
    
    quantile_user = user_rtgs_cnt.quantile([0.1, 0.25, 0.75, 0.9], axis = 0).drop(["user_id"], axis = 1)
    
    # # Removing the lower 10% of the outliers.
    # user_rtgs_cnt=user_rtgs_cnt[user_rtgs_cnt.rating_cnt>=quantile_user.iloc[0,0]]
    
    # These users are then removed from the dataset
    df = df.merge(user_rtgs_cnt[['user_id']],on="user_id", how="inner")   
    
    # Data preprocessing from item perspective
    
    # Count of Ratings per movie
    item_count = (df[["song_id","rating"]].groupby(['song_id']).count()).reset_index().rename(columns={"rating":"rating_per_item"})
    
    quantile_item=item_count.quantile([0.1,.25,.75,1], axis = 0).drop(["song_id"],axis=1)
    
    # # Removing all items which have less than 3 user counts i.e Q1 or based on a fixed number 
    # item_count = item_count[item_count.rating_per_item>=5].reset_index(drop=True)
    # item_count["item_subset"] = np.where(item_count.rating_per_item < quantile_item.iloc[2,0],1,2)
    item_count["item_subset"] = np.where(item_count.rating_per_item < quantile_item.iloc[2,0],1,2)
    
    # Data Sampling 
    
    sampled_ratings=pd.DataFrame()
    j = len(item_split)-1
    
    for i in item_count.item_subset.unique():
        sampled_ratings=sampled_ratings.append(item_count[item_count.item_subset==i]. \
                                               sample(n=int(item_split[j]*item_nos), random_state=10))
        j=j-1
        
    sampled_ratings.reset_index(drop=True, inplace=True)
    
    # Select user rows for only those movies which have been sampled
    df = df.merge(sampled_ratings[['song_id']],on="song_id", how="inner")
    
    # Since not all items are selected it may happen that we again get items with only user frequency.
    # Removing single frequency users so as to reduce sparsity and enable item-item comparison between pairs
    
    user_rtgs_cnt_2=(df.groupby(['user_id']).count()).iloc[:,0:1].reset_index().rename(columns={"song_id":"user_freq"})
    df = df.merge(user_rtgs_cnt_2,on="user_id", how="inner")
    
    # For any personalized recommendation to a user, we are setting a rule that user should have 
    # watched at least 7 movies then only make popular recommendations to him
    # df = df[df.user_freq>7] 
    df.drop(['user_freq'],axis=1, inplace=True)
    df = df.reset_index(drop=True)
    print("Total Number of Ratings in Sampled Dataset =", len(df))
    print("Total Number of Unique Users in Sample =", len(df.user_id.unique()))
    
    # Train-Test Split
    
    df_train = df.groupby(['user_id']).apply(lambda x : x.sample(frac=0.8,random_state=10)).reset_index(drop=True)
    z = df.merge(df_train,how='outer', on=['user_id','song_id','rating','timestamp'], indicator=True)
    df_test = z.query('_merge != "both"')
    df_test = df_test.drop(['_merge'],axis=1)
    df_test.reset_index(drop=True, inplace=True)
        
    return [df, df_train, df_test]

def build_weight_matrix(ratings):
    
    # define weight matrix
    w_matrix_columns = ['movie_1', 'movie_2', 'weight']
    w_matrix = pd.DataFrame(columns = w_matrix_columns)

    # calculate the similarity between pairs of movies
    unique_movies = np.unique(ratings['song_id'])
    print("Number of Unique Movies = ", len(unique_movies))

    for movie_1 in unique_movies:

        # extract all users who rated movie_1
        user_data = ratings[ratings['song_id'] == movie_1]
        unique_users = np.unique(user_data['user_id'])

        # record the ratings for users who rated both movie_1 and movie_2
        record_row_columns = ['user_id', 'movie_1', 'movie_2', 'rating_1', 'rating_2']
        record_movie_1_2 = pd.DataFrame(columns=record_row_columns)
        
        # for each customer C who rated movie_1 record the her ratings for movie_2 
        for c_userid in unique_users:
            c_movie_1_rating = user_data[user_data['user_id'] == c_userid]['rating'].iloc[0]
            # all movies of user c excluding movie_1
            c_user_data = ratings[(ratings['user_id'] == c_userid) & (ratings['song_id'] != movie_1)]
            c_unique_movies = np.unique(c_user_data['song_id'])

            # Iterate through all movies rated by customer C as movie=2
            for movie_2 in c_unique_movies:
               # the customer's rating for movie_2
                c_movie_2_rating = c_user_data[c_user_data['song_id'] == movie_2]['rating'].iloc[0]
                record_row = pd.Series([c_userid, movie_1, movie_2, c_movie_1_rating, c_movie_2_rating], 
                                       index=record_row_columns)
                record_movie_1_2 = record_movie_1_2.append(record_row, ignore_index=True)
        
        # computing the similarity between movie_1 and the other recorded movies tagged as movie_2
        unique_movie_2 = np.unique(record_movie_1_2['movie_2'])
        # going through each movie 2
        for movie_2 in unique_movie_2:
            paired_movie_1_2 = record_movie_1_2[record_movie_1_2['movie_2'] == movie_2]
            cosine_sim_numerator = (paired_movie_1_2['rating_1'] * paired_movie_1_2['rating_2']).sum()
            cosine_sim_denominator = np.sqrt(np.square(paired_movie_1_2['rating_1']).sum()) * \
                np.sqrt(np.square(paired_movie_1_2['rating_2']).sum())
                
            cosine_sim_denominator = cosine_sim_denominator if cosine_sim_denominator != 0 else 1e-8
            sim_value = cosine_sim_numerator / cosine_sim_denominator
            w_matrix = w_matrix.append(pd.Series([movie_1, movie_2, sim_value], index=w_matrix_columns), 
                                       ignore_index=True)
            
    #return the computed weight matrix
    return w_matrix

ratings, ratings_training, ratings_test = data_sampling(ratings_all, item_nos = 500)

start = time.time()
print('Building Weight Matrix - Item-Item Collaborative Filtering...')
w_matrix = build_weight_matrix(ratings_training)
print('Weight Matrix Successfully Built')
end = time.time()
print('\nTime Elapsed = '+str(end - start)+' secs')
"""
# Predict a rating for a given user and given movie
def predict(user_id, song_id, w_matrix, ratings):
    # predict the rating of the given movie by the given user
    user_other_ratings = ratings[ratings['user_id'] == user_id]
    user_unique_movies = np.unique(user_other_ratings['song_id'])
    sum_weighted_other_ratings = 0
    sum_weghts = 0
    for movie_j in user_unique_movies:
        # only calculate the weighted values when the weight between movie_1 and movie_2 exists in weight matrix
        w_movie_1_2 = w_matrix[(w_matrix['movie_1'] == song_id) & (w_matrix['movie_2'] == movie_j)]
        if len(w_movie_1_2) > 0:
            user_rating_j = user_other_ratings[user_other_ratings['song_id']==movie_j]
            sum_weighted_other_ratings += (user_rating_j['rating'].iloc[0] * w_movie_1_2['weight'].iloc[0])
            sum_weghts += np.abs(w_movie_1_2['weight'].iloc[0])

    # when sum_weights is 0 (in case there is no ratings from new users), use the mean ratings as 2.5
    if sum_weghts == 0:
        predicted_rating = 2.5
    else:
        predicted_rating = sum_weighted_other_ratings/sum_weghts
    predicted_rating = np.round(predicted_rating, 4)
    return predicted_rating

# Evaluate the learned recommender system on test data by converting the ratings to negative and positive
def rmse_eval(ratings_test, w_matrix, ratings_training):
    # predict all the ratings for test data
    ratings_test['prediction'] = pd.Series(np.zeros(ratings_test.shape[0]))
    
    for index, row_rating in ratings_test.iterrows():
        predicted_rating = predict(row_rating['user_id'], row_rating['song_id'], w_matrix, ratings_training)
        ratings_test.loc[index, 'prediction'] = predicted_rating
    
    rmse = np.round(np.sqrt(np.mean((ratings_test['prediction']-ratings_test['rating'])**2)), 4)
    mae = np.round(np.mean(np.abs(ratings_test['prediction']-ratings_test['rating'])), 4)
    ratings_test.drop(['prediction'], inplace=True, axis = 1)
    return rmse, mae

start = time.time()
print('Evaluating RMSE, MAE on test dataset...')
rmse, mae = rmse_eval(ratings_test, w_matrix, ratings_training)
print('RMSE on Test Dataset = ', rmse)
print('MAE on Test Dataset = ', mae)
end = time.time()
print('\nTime Elapsed = '+str(np.round(end - start, 4))+' secs')

# recommend top k movies for given user_id from movies that he/she has not seen
def recommend(userID, w_matrix, ratings, k=10):
    
    distinct_movies = np.unique(ratings['song_id'])
    user_rated_movies = np.unique(ratings[ratings['user_id']==userID]['song_id'])

    user_unrated_movies = pd.DataFrame(columns=['song_id', 'rating'])

    # predict the ratings for all movies that the user hasn't rated
    i = 0
    for movie in distinct_movies:
        if movie not in user_rated_movies:
            rating_value = predict(userID, movie, w_matrix, ratings)
            user_unrated_movies.loc[i] = [movie, rating_value]
            i = i + 1
        else:
            continue
            
    # select top k movies based on predicted ratings
    recommendations = user_unrated_movies.sort_values(by=['rating'], ascending=False).head(k)
    recommendations_list = [ [int(row['song_id']), row['rating']] for i,row in recommendations.iterrows() ]
    return recommendations_list

# taking top k recommendation for given list of users
def make_recommendation_for_users(users_list, ratings_training):
    users_recommendations_df = pd.DataFrame(columns=['user_id', 'recommendation'])
    count = 0
    for user in users_list:
        recommendations = recommend(user, w_matrix, ratings_training, k=10)
        users_recommendations_df.loc[count] = [user, recommendations]
        count+=1
        
    return users_recommendations_df

start = time.time()
print('Recommending 10 movies for all users...')
users_list_for_recommendation = list(set(ratings_training['user_id']) & set(ratings_test['user_id']))
users_recommendations_df = make_recommendation_for_users(users_list_for_recommendation, ratings_training)
print('Recommendations Successfully Generated')

end = time.time()
print('\nTime Elapsed = '+str(np.round(end - start, 4))+' secs')

users_recommendations_df.head()
 """

