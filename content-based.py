import Data
from Songs_Modeling import Song
from Users_Modeling import User
import similarity_measures as sim
import util
import pickle
import random as rnd
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import os
import random
import re

stopwords_list = set(stopwords.words('english'))
path_music_data= os.path.join('data', 'music database_count.csv')
dataframe = Data.read_songs_info('music_database_with_lists.csv')
usersframe = Data.read_users_songs_info('ratings.csv')
""" Repositories:
        Represented Items
        Feedback: reaction to items
"""

class ContentBasedRecommender:
    def __init__(self):
        self.content_simil = Content_analyzer()
        self.profiling = Profile_learner()
        self.filter = Filtering_component()
        # self.content_simil.extract_info()
        # self.tf_idf = sim._tf_x_idf(self.content_simil.title_list)
    
    # def _print_message(self, song:Song, recom_song):
    #     rec_items = len(recom_song)
        
    #     print(f'The {rec_items} recommended songs for {song} are:')
    #     for i in range(rec_items):
    #         print(f"Number {i+1}:")
    #         print(f"{recom_song[i][0]} with {round(recom_song[i][1], 3)} similarity score") 
    #         print("--------------------")
    
    def _print_message(self, user:User, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {user} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][0]} with {round(recom_song[i][1], 3)} similarity score") 
            print("--------------------")
    
    def recommend(self,user:User):
        s_sim_list = Filtering_component.recomend_to_user(user,self.content_simil.freq_matrix)
        for i in range(len(s_sim_list)):
            s_sim_list[i] = (self.content_simil.songs_list[s_sim_list[i][0]],s_sim_list[i][1])
        self._print_message(user,s_sim_list)

    def recommend_by_song(self, song:Song, to_recommend=10):
        recom = song.top_similar_songs(self.content_simil.songs_list, to_recommend, self.content_simil.freq_matrix)
        self._print_message(song, recom)
    

class Content_analyzer():
    """ Extract structured relevant information
        The main responsibility of the component is to represent the content of items  """
    def __init__(self):
        self.extract_info()
        self.freq_matrix = util.vectorize_songs(self.songs_list)
    def extract_info(self):
        self.songs_list = Song.from_Dataframe(dataframe)
        file = open('content-based system/songs_list.bin','wb')
        pickle.dump(self.songs_list,file)
        file.close
        self.title_list = [s.title for s in self.songs_list]

class Profile_learner():
    """ Collects data representative of the user preferences and tries to generalize this data, 
        in order to construct the user profile.  """
    def __init__(self,song_list=None):
        self.profile_users()
        self.vector_by_user(song_list)

    def profile_users(self):
        self.users_rates = User.rate_from_Dataframe(usersframe)
        # try:
        #     uf = open('content-based system/user_list.bin','rb')
        #     self.users_list = pickle.load(uf)
        #     uf.close()
        # except:
        self.users_list = User.users_from_Dataframe(self.users_rates)
            # pass
        file = open('content-based system/user_list.bin','wb')
        pickle.dump(self.users_list, file)
        file.close()
    def vector_by_user(self,song_list):
        for user in self.users_list:
            user:User
            user.vectorize_user(song_list)
        
        self.vectors_by_user = [ u.vector for u in self.users_list]

    def artists_to_recommend(self, user_id:int, top:int):
        user:User = self.users_list[user_id - 1]
        ret = user.top_prefered_artists(dataframe,top)
        return ret
    def genres_to_recommend(self, user_id:int, top:int):
        user:User = self.users_list[user_id - 1]
        ret = user.top_prefered_genres(dataframe,top)
        return ret
    def new_user(self):
        pass

class Filtering_component():
    """ 
        Exploits the user profile to suggest relevant items by matching the profile representation against that of items to be recommended
        The result is a continuous relevance judgment, resulting in a ranked list of potentially interesting items.
    """

    def recomend_to_user(user:User, songs_freq_matrix:np.ndarray):
        songs_similarity = Filtering_component.relevant_songs(user,songs_freq_matrix)
        reduced_similarity = Filtering_component.remove_rated_songs(user,songs_similarity)
        to_recom = user.songs_to_recommend
        # reduced_similarity = list(reduced_similarity.values())[:to_recom*3]
        result = []

        rnd.seed(time.time())
        i = 0
        c = to_recom*3
        while i < to_recom:
            choice = rnd.randint(0,c-1)
            if not reduced_similarity.get(choice):
                continue
            s = list(reduced_similarity.keys())[choice]
            result.append((s, reduced_similarity[choice]))

            reduced_similarity.pop(choice)
            i+=1
            c-=1
        return result

    def remove_rated_songs(user:User, songs_similarity:dict):
        for s_id in user.rates:
            songs_similarity.pop(s_id-1)
        return songs_similarity

    def relevant_songs(user:User, songs_freq_matrix:np.ndarray):
        simil = sim.Similarity()
        user_song_sim = {s:0 for s in range(songs_freq_matrix.shape[1])}
        for s in range(songs_freq_matrix.shape[1]):
            # song_col = 
            user_song_sim[s] = simil.cosine_similarity(songs_freq_matrix[:,s],user.vector)
        user_song_sim = dict(sorted(user_song_sim.items(), key=lambda item: item[1], reverse=True))
        return user_song_sim

cb = ContentBasedRecommender()
u = cb.profiling.users_list[7]
# cb.recommend(u)
# print(cb.profiling.artists_to_recommend(3,5))
# print(cb.profiling.genres_to_recommend(3,5))