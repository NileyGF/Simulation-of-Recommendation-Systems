from Core import *
from Songs_Modeling import Song
from Users_Modeling import User
import Agents as agent
import Agents_actions as act
import similarity_measures as sim
import util
import pickle
import random as rnd
import time
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

# stopwords_list = set(stopwords.words('english'))
# path_music_data= os.path.join('data', 'music database_count.csv')
# dataframe = Data.read_songs_info('music_database_with_lists.csv')
# usersframe = Data.read_users_songs_info('ratings.csv')


class ContentBasedRecommender(Recommender):
    def __init__(self,songsFrame:pd.DataFrame,userFrame:pd.DataFrame):
        self.content_simil = Content_analyzer(songsFrame)
        self.profiling = Profile_learner(userFrame,self.content_simil.songs_list)
        self.filter = Filtering_component(self.content_simil.songs_list)
        self.user_conv = {}
    def get_songs(self):
        return self.content_simil.songs_list
    def new_user(self,agent:agent.Agent):
        user_id = agent.id
        id = self.user_conv.get(user_id)
        if not id:
            self.user_conv[user_id] = self.profiling.next_id
            self.profiling.next_id += 1
        return self.profiling.new_user(agent,self.user_conv[user_id])

    def _print_message(self, user:User, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {user} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][0]} with {round(recom_song[i][1], 3)} similarity score") 
            print("--------------------")
    
    def recommend(self,ag:agent.Agent):
        user = self.profiling.users_dict[self.user_conv[ag.id]]
        recommendation = self.filter.recomend_to_user(user,self.content_simil.freq_matrix)
        # recommendation = s_sim_list.copy()
        # for i in range(len(recommendation)):
        #     recommendation[i] = recommendation[i][1]
        # self._print_message(user,s_sim_list)
        ratings, changed = ag.received_recommendation(recommendation)
        for i in range(len(recommendation)):
            self.profiling.new_rate(user, recommendation[i], ratings[i])
        self.profiling.revector(user)
        return changed

    def recommend_by_song(self, song:Song, to_recommend=10):
        recom = song.top_similar_songs(self.content_simil.songs_list, to_recommend, self.content_simil.freq_matrix)
        self._print_message(song, recom)
    
class Content_analyzer():
    """ Extract structured relevant information
        The main responsibility of the component is to represent the content of items  """
    def __init__(self,songsFrame:pd.DataFrame):
        self.songsFrame = songsFrame
        self.extract_info()
        self.freq_matrix = util.vectorize_songs(self.songs_list)
    
    def extract_info(self):
        self.songs_list = Song.from_Dataframe(self.songsFrame)
        file = open('content-based data/songs_list.bin','wb')
        pickle.dump(self.songs_list,file)
        file.close
        self.title_list = [s.title for s in self.songs_list]

class Profile_learner():
    """ Collects data representative of the user preferences and tries to generalize this data, 
        in order to construct the user profile.  """
    def __init__(self,userFrame:pd.DataFrame,song_list=None):
        self.usersframe = userFrame
        self.song_l = song_list
        self.profile_users()
        self.vector_by_user(song_list)

    def profile_users(self):
        self.users_rates = User.rate_from_Dataframe(self.usersframe)
        # try:
        #     uf = open('content-based data/user_list.bin','rb')
        #     self.users_dict = pickle.load(uf)
        #     uf.close()
        # except:
        self.users_dict = User.users_from_Dataframe(self.users_rates)
            # pass
        self.next_id = max(list(self.users_rates.keys())) + 1
        file = open('content-based data/user_list.bin','wb')
        pickle.dump(self.users_dict, file)
        file.close()
    
    def vector_by_user(self,song_list):
        self.vectors_by_user = []
        for user_id in self.users_dict:
            user:User = self.users_dict[user_id]
            user.vectorize_user(song_list)        
            self.vectors_by_user.append(user.vector)
        return self.vectors_by_user

    """  def artists_to_recommend(self, user_id:int, top:int):
        user:User = self.users_dict[user_id]
        ret = user.top_prefered_artists(dataframe,top)
        return ret
    def genres_to_recommend(self, user_id:int, top:int):
        user:User = self.users_dict[user_id]
        ret = user.top_prefered_genres(dataframe,top)
        return ret """
    def revector(self,user:User):
        user.vectorize_user(self.song_l)
        self.vectors_by_user.append(user.vector)

    def new_user(self,agent:agent.Agent,user_id):
        user = User(user_id)
        regist:act.Action = agent.register_in_system()
        if regist is act.RegisterAction:
            regist.register_explicit(user)
        user.vectorize_user(self.song_l)
        self.users_dict[user.id] = user
        self.vectors_by_user.append(user.vector)
        return user

    def new_rate(self,user:User,song:Song,rating:int):
        rated = self.users_rates.get(user.id)
        if not rated:
            self.users_rates[user.id] = [(song.id,rating)]
        else:
            self.users_rates[user.id].append( (song.id,rating) )
        user.add_rate(song.id,rating)

class Filtering_component():
    """ 
        Exploits the user profile to suggest relevant items by matching the profile representation against that of items to be recommended
        The result is a continuous relevance judgment, resulting in a ranked list of potentially interesting items.
    """
    def __init__(self,songs_list:list):
        self.top_popular_songs(songs_list)

    def recomend_to_user(self, user:User, songs_freq_matrix:np.ndarray):
        songs_similarity = self.relevant_songs(user,songs_freq_matrix)
        reduced_similarity = self.remove_rated_songs(user,songs_similarity)
        reduced_similarity = self.remove_zero_songs(reduced_similarity)
        to_recom = user.songs_to_recommend
        
        # if there are not enough similars songs, add the most populars ones
        if len(reduced_similarity) < to_recom:
            i = len(reduced_similarity)
            for song in self.top_pop_songs:
                reduced_similarity[i] = song
                i+=1       
            to_recom = min(to_recom,len(reduced_similarity)) 

        result = []
        rnd.seed(time.time())
        i = 0
        c = to_recom*3
        while i < to_recom:
            choice = rnd.randint(0,c-1)
            s = reduced_similarity.get(choice)
            if s is None:
                continue
            result.append(reduced_similarity[choice])
            reduced_similarity.pop(choice)
            i+=1
            c-=1
        return result

    def remove_rated_songs(self, user:User, songs_similarity:dict):
        for s_id in user.rates:
            songs_similarity.pop(s_id-1)
        return songs_similarity

    def remove_zero_songs(self, songs_similarity:dict):
        result = {}
        i = 0 
        for s in songs_similarity:
            if songs_similarity[s] > 0:
                result[i] = songs_similarity[s]
                i+=1
        return result

    def relevant_songs(self, user:User, songs_freq_matrix:np.ndarray):
        simil = sim.Similarity()
        user_song_sim = {s:0 for s in range(songs_freq_matrix.shape[1])}
        for s in range(songs_freq_matrix.shape[1]):
            user_song_sim[s] = simil.cosine_similarity(songs_freq_matrix[:,s],user.vector)
        user_song_sim = dict(sorted(user_song_sim.items(), key=lambda item: item[1], reverse=True))
        return user_song_sim
        
    def top_popular_songs(self, songs_list:list, top:int=20):
        sorted_by_listened = sorted(songs_list, key=lambda item: item.listen_count, reverse=True)
        
        while top < len(sorted_by_listened)-1 and sorted_by_listened[top].listen_count == sorted_by_listened[top+1].listen_count:
            top += 1
        self.top_pop_songs = sorted_by_listened[:top]
        

# cb = ContentBasedRecommender()
# u = cb.profiling.users_dict[8]
# cb.recommend(u)
# print(cb.profiling.artists_to_recommend(3,5))
# print(cb.profiling.genres_to_recommend(3,5))