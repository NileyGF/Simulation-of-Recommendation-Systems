
import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
import time
import csv

def read_songs_info(file='music_database_with_lists.csv'):
    path_music_data= os.path.join('data', file)
    songs_csv = pd.read_csv(path_music_data)
    songs_csv.columns = ['song_id', 'title', 'number of artists', 'artists', 'genres', 'Release Year', 'listen_count']

    songs_csv["artists"] = songs_csv["artists"].apply(eval)
    songs_csv["genres"] = songs_csv["genres"].apply(eval)
    print(f"There are {songs_csv.shape} songs in the dataset")
    unique_songs = songs_csv['song_id'].unique()
    print(f"There are {unique_songs.shape[0]} unique songs in the dataset")
    return songs_csv

def read_users_songs_info(file='ratings.csv'):
    path_ratings_data= os.path.join('data', file)
    ratings_csv = pd.read_csv(path_ratings_data)
    ratings_csv.columns = ['user_id', 'song_id', 'rating']

    print(f"There are {ratings_csv.shape} ratings in the dataset")
    unique_users = ratings_csv['user_id'].unique()
    print(f"There are {unique_users.shape[0]} unique users in the dataset")
    return ratings_csv

def songs_by_user(dataframe:pd.DataFrame):
    song_user = dataframe.groupby('user_id')['song_id'].count()
    plt.figure(figsize=(16, 8))
    sns.distplot(song_user.values, color='orange')
    # plt.hist(song_user.values)
    plt.show()
    plt.savefig('songs_by_user', format='png')

def top_popular_artists(songs_data:pd.DataFrame, Top:int):
    """ top popular artists """
    # songs_data["artists"] = songs_data["artists"].apply(eval)
    artists_popularity = {}
    for index, row in songs_data.iterrows():
        for art in row['artists']:
            ap = artists_popularity.get(art)
            if not ap:
                artists_popularity[art] = int(row['listen_count'])
            else:
                artists_popularity[art] += int(row['listen_count'])
        
    top_pop_artists = dict(sorted(artists_popularity.items(), key=lambda item: item[1], reverse=True))

    labels = list(top_pop_artists.keys())
    counts = list(top_pop_artists.values())
    while Top < len(counts) and counts[Top] == counts[Top+1]:
        Top += 1
    labels = labels[:Top]
    counts = counts[:Top]
    labels.reverse()
    counts.reverse()

    plt.figure()
    plt.barh(labels,counts)
    plt.show()
    plt.savefig('top_artists', format='png')
    pass

def top_popular_genres(songs_data:pd.DataFrame, Top:int):
    """ top popular genres """
    # songs_data["genres"] = songs_data["genres"].apply(eval)
    genres_popularity = {}
    for index, row in songs_data.iterrows():
        for gen in row['genres']:
            ap = genres_popularity.get(gen)
            if not ap:
                genres_popularity[gen] = int(row['listen_count'])
            else:
                genres_popularity[gen] += int(row['listen_count'])
        
    top_pop_genres = dict(sorted(genres_popularity.items(), key=lambda item: item[1], reverse=True))

    labels = list(top_pop_genres.keys())
    counts = list(top_pop_genres.values())
    while Top < len(counts) and counts[Top] == counts[Top+1]:
        Top += 1
    labels = labels[:Top]
    counts = counts[:Top]
    labels.reverse()
    counts.reverse()

    plt.figure()
    plt.barh(labels,counts)
    plt.show()
    plt.savefig('top_genres', format='png')
    pass

def top_popular_songs(songs_data:pd.DataFrame, Top:int):
    """ top popular songs """
    top_pop_songs = songs_data.sort_values(by=['listen_count'], ascending=False)
    # top_pop_songs = top_pop_songs.reset_index().sort_values(by=['listen_count'])
    # top_pop_songs = songs['listen_count'].count().reset_index().sort_values(['listop_count', 'Song Title','Artist'])
    top_pop_songs['percentage'] = round(top_pop_songs['listen_count'].div(top_pop_songs['listen_count'].sum())*100, 2)
    while Top < top_pop_songs.shape[0] and top_pop_songs.iat[Top,6] == top_pop_songs.iat[Top+1,6]:
        Top += 1
    top_pop_songs = top_pop_songs[:Top]

    labels = top_pop_songs['title'].tolist()
    counts = top_pop_songs['listen_count'].tolist()
    labels.reverse()
    counts.reverse()

    plt.figure()
    plt.barh(labels,counts)
    # plt.ylim(150,171)
    plt.show()
    plt.savefig('top_songs', format='png')
    pass

def artists_to_list(songs_data):
    newcol = []
    for index, row in songs_data.iterrows():
        artists = row['artists']
        no_a = row['number of artists']
        newcol.append(get_list(artists, no_a))
    return newcol

def genres_to_list(songs_data):
    newcol = []
    for index, row in songs_data.iterrows():
        genres = row['genres']
        newcol.append(get_list(genres, -1))
    return newcol

def get_list(string:str, len_res):
    if(len_res == 1):
        return [string.strip()]
    res = []
    sep = string.split(',')
    for i in sep:
        i+= ' '
        res.append(i.strip())
    return res

def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

# rnd.seed(time.time())
# songs = read_songs_info('music_database_with_lists.csv')
# listen_count = np.ndarray(dtype=int,shape=songs.shape[0])
# for i in range(listen_count.shape[0]):
#     r1 = rnd.gauss(100,60)
#     if r1 < 0:
#         r1 = 0
#     listen_count[i] = int(r1)

# songs['listen_count'] = listen_count
# songs.to_csv('data/music_database_with_lists.csv',index=False)

# def random_rate(number_users, number_songs, number_songs_to_rate):
#     ratings = []
#     for i in range(number_users):
#         songs_to_rate = int(rnd.randint(0,number_songs_to_rate-1))
#         rated_songs = [] # to avoid the situation where a user rate the same song more than once
#         for j in range(number_songs_to_rate - songs_to_rate):
#             rate = rnd.randint(0,10)
#             if rate <= 1: rate = 0
#             song_id = int(rnd.uniform(1,number_songs + 1))
#             while (song_id in rated_songs) or (song_id > number_songs):
#                 song_id = rnd.randint(1,number_songs + 1)
#             rated_songs.append(song_id)
#             ratings.append([i+1, song_id, rate*0.5])    
#     return ratings

# rate_fields = ['user_id', 'song_id', 'rating']
# rate_list = random_rate(40, 2414, 25) 
# ratings_f = open('data/ratings.csv', 'w+')
# writecsv = csv.writer(ratings_f)
# writecsv.writerow(rate_fields)
# writecsv.writerows(rate_list)
# ratings_f.close()

# Ten top popular artists
""" ten_pop_artists = songs.groupby(['artist_name'])['listen_count'].count().reset_index().sort_values(
    ['listen_count', 'artist_name'], ascending=[0,1])
ten_pop_artists = ten_pop_artists[:10]

plt.figure()
labels = ten_pop_artists['artist_name'].toList()
counts = ten_pop_artists['listen_count'].toList()
sns.barplot(x=counts, y=labels, palette='Set2')
sns.despine(left=True, bottom=True)

#  
song_user = songs.groupby('user_id')['song_id'].count()

plt.figure(figsize=(16,8))
sns.displot(song_user.values, color='orange')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show();

print(f"A usesr listens to an average of {np.mean(song_user)} songs")
print(f"A usesr listens to an average of {np.median(song_user)} songs, with a minimum {np.min(song_user)} and maximum {np.max(song_user)} songs")

#users who have listened to at least 16 songs
song_ten_id = song_user[song_user > 16].index.to_list()
df_song_id_more_ten = songs[songs['user_id'].isin(song_ten_id)].reset_index(drop=True)

#convert the data frame into a pivot table
df_songs_features = df_song_id_more_ten.pivot(index='song_id', columns='user_id', values='listen_count').fillna(0)
mat_songs_features = csr_matrix(df_songs_features.values)
df_unique_songs = songs.drop_duplicates(subset=['song_id']).reset_index(drop=True)[['song_id', 'title']]
decode_id_song = {
    song: i for i, song in enumerate(list(df_unique_songs.set_index('song_id').loc[df_songs_features.index]))
}
 """

# dataframe = read_songs_info('music_database_with_lists.csv')
# usersframe = read_users_songs_info()
# songs_by_user(usersframe)
# top_popular_songs(dataframe, 16)
# top_popular_artists(dataframe, 13)
# top_popular_genres(dataframe, 13)
# dataframe['artists'] = artists_to_list(dataframe)
# dataframe['genres'] = genres_to_list(dataframe)
# dataframe.to_csv('data/music_database_with_lists.csv',index=False)
