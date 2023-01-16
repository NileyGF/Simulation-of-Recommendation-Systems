import numpy as np
import pandas as pd
from enum import Enum
import util

class Listening_behavior(Enum):
    Indiferents = 1     #  7% # They would not lose much sleep if music ceased to exist, they are a predominant type of listeners of the whole population.
    Casuals = 2         # 21% # Music plays a welcome role, but other things are far more important
    Enthusiasts = 3     # 32% # Music is a key part of life but is also balanced by other interests.
    Savants = 4         # 40% # Everything in life seems to be tied up with music. Their musical knowledge is very extensive.
# number_of_songs_to_recommend_per_listening_behavior
ns_lb = {Listening_behavior.Indiferents:    4,
         Listening_behavior.Casuals:        7,
         Listening_behavior.Enthusiasts:    10,
         Listening_behavior.Savants:        15}

class User:
    def __init__(self, id:int):
        self.id = id
        self.rates = {}
        self.impl_artists = None
        self.expl_artists = None
        self.impl_genres = None
        self.expl_genres = None
        self.prefered_artists = None
        self.prefered_genres = None     
        self.vector = None
        self.songs_to_recommend = 5
    
    def __str__(self) -> str:
        string = "User " + str(self.id) 
        string += "\n rates: \n" + str(self.rates)
        return string
    def __repr__(self) -> str:
        return self.__str__()

    def add_rate(self, song_id:int, rate:float):
        song_id = int(song_id)
        self.rates[song_id] = rate
    
    def vectorize_user(self, song_list:list):
        if util.Util.info_by_song == None or util.Util.sorted_vocab == None:
            terms_by_title, artists_by_song, genres_by_song, titles_vocabulary, artists, genres = util.process_song_list(song_list)
        else:
            terms_by_title = util.Util.info_by_song[0]
            artists_by_song = util.Util.info_by_song[1]
            genres_by_song = util.Util.info_by_song[2]
            titles_vocabulary = util.Util.sorted_vocab[0]
            artists = util.Util.sorted_vocab[1]
            genres = util.Util.sorted_vocab[2]

        id_ind = util.get_Id_Ind_dict(song_list)

        vector = np.zeros(shape=(len(titles_vocabulary) + len(artists) + len(genres)), dtype=float)
        if len(self.rates) == 0:
            self.vector = vector
            return vector

        for i in range(len(titles_vocabulary)):
            freq_i = 0
            for s_id in self.rates:
                freq_i += terms_by_title[id_ind[s_id]].count(titles_vocabulary[i])
            vector[i] = freq_i
        ind = len(titles_vocabulary)

        for s_id in self.rates:
            for art in artists_by_song[id_ind[s_id]]:
                i = artists.index(art)
                vector[ind+i] +=1
        ind = len(titles_vocabulary) + len(artists)

        for s_id in self.rates:
            for gen in genres_by_song[id_ind[s_id]]:
                i = genres.index(gen)
                vector[ind+i] +=1
        self.vector = vector
        return vector

    def set_static_info(self,listening_behavior:Listening_behavior,prefered_artists:list,prefered_genres:list):
        self.listening_behavior = listening_behavior
        self.songs_to_recommend = ns_lb[listening_behavior]
        if isinstance(prefered_artists,list):
            self._explicit_prefered_artist(prefered_artists)
        if isinstance(prefered_genres,list):
            self._explicit_prefered_genres(prefered_genres)
        
    def rate_from_Dataframe(DataFrame:pd.DataFrame):
        unique_users = DataFrame['user_id'].unique()
        ret = {id:[] for id in unique_users}
        for index, row in DataFrame.iterrows():
            id = row['user_id']
            ret[id].append( (int(row['song_id']),row['rating']) )
        return ret

    def users_from_Dataframe(rate_dict:dict):
        ret = {}
        for user_id in rate_dict:
            user = User(user_id)
            for rating in rate_dict[user_id]:
                user.add_rate(rating[0], rating[1])
            ret[user_id] = user
        return ret
    
    def top_rated_songs_ids(self, top:int=10,all=True):
        top_songs = self.rates.copy()
        top_songs = dict(sorted(top_songs.items(), key=lambda item: item[1], reverse=True))
        if all: return top_songs
        try: top = int(top)
        except: top = 10
        i=0
        while list(top_songs.values())[i] > 0:
            i+=1
            if i >=top: break
        if i < top: top = i

        return slice_dict(top_songs, 0, top)

    def top_prefered_artists(self, DataFrame:pd.DataFrame, top:int = None):
        if top != None:
            try: top = int(top)
            except: top = None
        if not self.impl_artists:
            self._implicit_prefered_artist(DataFrame)
        if not self.expl_artists:
            #use only implicit data
            self.prefered_artists = {key:round(self.impl_artists[key],4) for key in self.impl_artists}
        else:
            #use implicit data and explicit data
            for art in self.impl_artists:
                expl_rate = self.expl_artists.get(art)
                if not expl_rate:
                    self.prefered_artists[art] = round(self.impl_artists[art],4)
                else:
                    if self.impl_artists[art] == 0:
                        self.prefered_artists[art] = self.expl_artists[art]
                    else:
                        # not explicit nor implicit are 0
                        self.prefered_artists[art] = round( 0.4 * self.impl_artists[art] + 0.6 * self.expl_artists[art], 4)
        self.prefered_artists = dict(sorted(self.prefered_artists.items(), key=lambda item: item[1], reverse=True))
        # Return only non-0-relevance ones
        if top == None: 
            return self.prefered_artists
        i=0
        while list(self.prefered_artists.values())[i] > 0:
            i+=1
            if i >=top: break
        if i < top: top = i

        return slice_dict(self.prefered_artists,0,top)
    def top_prefered_genres(self, DataFrame:pd.DataFrame, top:int = None):
        if top != None:
            try: top = int(top)
            except: top = None
        if not self.impl_genres:
            self._implicit_prefered_genres(DataFrame)
        if not self.expl_genres:
            #use only implicit data
            self.prefered_genres = {key:round(self.impl_genres[key],4) for key in self.impl_genres}
        else:
            #use implicit data and explicit data
            for gen in self.impl_genres:
                expl_rate = self.expl_genres.get(gen)
                if not expl_rate:
                    self.prefered_genres[gen] = round(self.impl_genres[gen],4)
                else:
                    if self.impl_genres[gen] == 0:
                        self.prefered_genres[gen] = self.expl_genres[gen]
                    else:
                        #not explicit nor implicit are 0
                        self.prefered_genres[gen] = round(0.4 * self.impl_genres[gen] + 0.6 * self.expl_genres[gen], 4)
        self.prefered_genres = dict(sorted(self.prefered_genres.items(), key=lambda item: item[1], reverse=True))
        # Return only non-0-relevance ones
        if top == None: 
            return self.prefered_genres
        i=0
        while list(self.prefered_genres.values())[i] > 0:
            i+=1
            if i >=top: break
        if i < top: top = i
        
        return slice_dict(self.prefered_genres,0,top)

    def _implicit_prefered_artist(self, DataFrame:pd.DataFrame):
        self.impl_artists = self.implicit_category_profile(DataFrame,'artists')    
    def _explicit_prefered_artist(self, artists_list:list):
        self.expl_artists = {art:4.5 for art in artists_list}
    
    def _implicit_prefered_genres(self, DataFrame:pd.DataFrame):
        self.impl_genres = self.implicit_category_profile(DataFrame,'genres')    
    def _explicit_prefered_genres(self, genres_list:list):
        self.expl_genres = {gen:4.5 for gen in genres_list}
    
    def implicit_category_profile(self, songsFrame:pd.DataFrame, category:str):
        category_profile = {}
        for index, row in songsFrame.iterrows():
            s_id = row['song_id']
            rate = self.rates.get(s_id)
            if not rate: rate = 0
            cat_list = row[category]
            for cat in cat_list:
                so_far = category_profile.get(cat)
                if not so_far:
                    category_profile[cat] = (1,rate) 
                else: 
                    category_profile[cat] = (category_profile[cat][0] + 1, category_profile[cat][1] + rate)
        for cat in category_profile:
            #i would like to change the normalization to divide for the number of songs with that cat.
            category_profile[cat] = category_profile[cat][1] / category_profile[cat][0]
        return category_profile
    
def slice_dict(dictionary:dict, from_ind:int, to_ind:int):
    result = {}
    i = 0
    for key in dictionary:
        if i < from_ind: continue
        if i >= to_ind: break
        result[key] = dictionary[key]
        i += 1
    return result

