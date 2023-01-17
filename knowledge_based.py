from Songs_Modeling import Song
from Users_Modeling import User, Listening_behavior
from CSP import *
from Core import *
import util
import similarity_measures as sim
import Agents_actions as act
import time
import pickle
import pandas as pd
import random as rnd

class Knowledge_based_recommender(Recommender):
    def __init__(self,songsFrame:pd.DataFrame,userFrame:pd.DataFrame):
        self.songsFrame = songsFrame
        self.extract_song_info()
        self.top_popular_songs = sorted(self.songs_list, key=lambda item: item.listen_count, reverse=True)
        self.usersframe = userFrame
        self.profile_users()
        self.vector_by_user()
        self.user_conv = {}
    def get_songs(self):
        return self.songs_list

    def profile_users(self):
        self.users_rates = User.rate_from_Dataframe(self.usersframe)
        # try:
        #     uf = open('knowledge-based data/user_list.bin','rb')
        #     self.users_dict = pickle.load(uf)
        #     uf.close()
        # except:
        self.users_dict = User.users_from_Dataframe(self.users_rates)
            # pass
        self.next_id = max(list(self.users_rates.keys())) + 1
        file = open('knowledge-based data/user_list.bin','wb')
        pickle.dump(self.users_dict, file)
        file.close()
    
    def vector_by_user(self):
        self.vectors_by_user = []
        for user_id in self.users_dict:
            user:User = self.users_dict[user_id]
            user.vectorize_user(self.songs_list)        
            self.vectors_by_user.append(user.vector)
        return self.vectors_by_user
    def revector(self,user:User):
        user.vectorize_user(self.songs_list)
        self.vectors_by_user.append(user.vector)

    def new_user(self,agent:agent.Agent):
        ag_id = agent.id
        user_id = self.user_conv.get(ag_id)
        if not user_id:
            self.user_conv[ag_id] = self.next_id
            self.next_id += 1
            user_id = self.user_conv[ag_id]
        else:
            print("It's not a new user")
            return None
            
        user = User(user_id)
        regist:act.Action = agent.register_in_system()
        if regist is act.RegisterAction:
            regist.register_explicit(user)
        user.vectorize_user(self.songs_list)
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
    
    def extract_song_info(self):
        self.songs_list = Song.from_Dataframe(self.songsFrame)
        self.freq_matrix = util.vectorize_songs(self.songs_list)
        file = open('knowledge-based data/songs_list.bin','wb')
        pickle.dump(self.songs_list,file)
        file.close
        self.__ids_songs= {}
        for s in self.songs_list:
            self.__ids_songs[s.id] = s

    def get_songs_from_ids(self,ids_list):
        songs = []
        for id in ids_list:
            songs.append(self.__ids_songs[id])
        return songs
    
    def recommend(self, ag: agent.Agent):
        user:User = self.users_dict[self.user_conv[ag.id]]
        recommendation = self.csp_recommend(user,None,None)
        for i in range(len(recommendation)):
            recommendation[i] = recommendation[i][1]
        ratings, changed = ag.received_recommendation(recommendation)
        for i in range(len(recommendation)):
            self.new_rate(user, recommendation[i], ratings[i])
        self.revector(user)
        return changed
        
    def csp_recommend(self, user:User,median_listen_count:int,median_rates:list):
        self.current_user_vector = user.vector
        self.current_repetitions = user.songs_to_recommend
        self.current_pref_artists = user.prefered_artists
        self.current_pref_genders = user.prefered_genres
        self.current_rated_songs_sorted = user.top_rated_songs_ids(all=True)
        self.current_rated_songs_sorted = self.get_songs_from_ids(self.current_rated_songs_sorted.keys())
        if user.listening_behavior != None:
            self.listening_behavior = user.listening_behavior
        else:
            self.listening_behavior = Listening_behavior.Casuals
        song_vars = []
        for i in range(self.current_repetitions):
            song_vars.append(str("song_"+str(i)))
        domain = self.songs_list.copy()
        song_vars = Variable.from_names_to_equal_domain(song_vars,domain)
        
        # constraint 1: if user is Indiferent, the songs must be similar to liked_songs(0.1), or a popular one (top 100)
        # constraint 1: if user is Casual, the songs must be similar to liked_songs(0.3), or a popular one (top 70)
        # constraint 1: if user is Enthusiast, the songs must be similar to liked_songs(0.5), or a popular one (top 50)   
        # constraint 1: if user is Savant, the songs must be similar to liked_songs(0.7), or a popular one (top 20)
        const1_vars = []  
        for k in song_vars:
            const1_vars.append(song_vars[k])
        const1 = Constraint(const1_vars, self.__evaluate_similar_songs)
        # constraint 2: if user is Indiferent, the song must have a liked genre or artist (probability of bypassing this = 0.75)
        # constraint 2: if user is Casual, the song must have a liked genre or artist (probability of bypassing this = 0.6)
        # constraint 2: if user is Enthusiast, the songs must have a liked genre or artist (probability of bypassing this = 0.45) 
        # constraint 2: if user is Savant, the songs must have a liked genre or artist (probability of bypassing this = 0.3)
        const2_vars = []   
        for k in song_vars:
            const2_vars.append(song_vars[k])
        const2 = Constraint(const2_vars, self.__evaluate_artists_genres)
        # constraint 3: the recommendations cannot be in user.rated_songs
        const3_vars = []   
        for k in song_vars:
            const3_vars.append(song_vars[k])
        const3 = Constraint(const3_vars, self.__evaluate_not_repeated)
        # constraint 4: the recommendations must be differents
        const4_vars = []
        for k in song_vars:
            const4_vars.append(song_vars[k])
        const4 = Constraint(const4_vars, self.__evaluate_all_diff)
        constraints = (const1, const2, const3, const4)
        recommendation_problem = ConstraintProblem(constraints)
        var_collection = self.get_csp_solution(constraints,recommendation_problem)
        return var_collection
    
    def get_csp_solution(self,constraints:tuple,recommendation_problem:ConstraintProblem):
        start_time = time.process_time()
        solution = classic_heuristic_backtracking_search(recommendation_problem, with_history=True)
        end_time = time.process_time()
        time_results = [round(end_time - start_time,4)]
        histories_lengths = []
        unsatisfied_constraints_amounts = []
        histories_lengths.append(len(solution))
        unsatisfied_constraints_amounts.append(len(recommendation_problem.get_unsatisfied_constraints()))

        constraint_solver = "classic_heuristic_backtracking_search"
        print("#" * 80)

        print("displaying performance results of solver: '", constraint_solver, sep='')
        overall_constraints_amount = str(len(recommendation_problem.get_constraints()))
        print("unsatisfied_constraints_amounts out of", overall_constraints_amount, "overall constraints:",
                unsatisfied_constraints_amounts)
        if hasattr(solution, "__len__"):
            print("solution lengths (number of assignment and unassignment actions):", histories_lengths)

        print("time results (seconds):", time_results)
        return solution

    def __evaluate_similar_songs(self,vars:tuple) -> bool:
        """ the songs must be similar to liked songs """
        margen_dict = { Listening_behavior.Indiferents:    0.1,
                        Listening_behavior.Casuals:        0.3,
                        Listening_behavior.Enthusiasts:    0.5,
                        Listening_behavior.Savants:        0.7}
        popular_dict = { Listening_behavior.Indiferents:   100,
                        Listening_behavior.Casuals:         70,
                        Listening_behavior.Enthusiasts:     50,
                        Listening_behavior.Savants:         20}
        
        simil_list = []
        simil = sim.Similarity()
        for v in vars:
            v:Variable
            song:Song = v.value
            simil_list.append(simil.cosine_similarity(song.vector,self.current_user_vector))
        margen = margen_dict[self.listening_behavior]
        main_condition = False
        for s in simil_list:            
            main_condition = main_condition or s >= margen
        if main_condition:
            return True

        top = popular_dict[self.listening_behavior]
        top_popular = self.top_popular_songs[:top]
        for v in vars:
            v:Variable
            song:Song = v.value
            if song in top_popular:
                return True
        return False
    def __evaluate_artists_genres(self,vars:tuple) -> bool:
        """ the songs must have a liked genre or artist """
        if self.current_pref_artists == None or self.current_pref_genders == None:
            return True
        bypass_dict = { Listening_behavior.Indiferents:    0.75,
                        Listening_behavior.Casuals:        0.6,
                        Listening_behavior.Enthusiasts:    0.45,
                        Listening_behavior.Savants:        0.3}
        bypass = bypass_dict[self.listening_behavior]
        simil_arts_list = []
        simil_genrs_list = []
        for v in vars:
            v:Variable
            song:Song = v.value
            intersection_cardinality = len(set.intersection(*[set(song.artists), set(self.current_pref_artists)]))
            simil_arts_list.append(intersection_cardinality)
            intersection_cardinality = len(set.intersection(*[set(song.genres), set(self.current_pref_genders)]))
            simil_genrs_list.append(intersection_cardinality)
        main_condition = False
        for s in vars:
            main_condition = main_condition or ( simil_arts_list[s] >= 1 or simil_genrs_list[s] >=1)
        if main_condition:
            return True
        return rnd.choices([True,False],[bypass,1-bypass]) [0]

    def __evaluate_not_repeated(self,vars:tuple) -> bool:
        """ the recommendations cannot be in user.rated_songs """
        if self.current_rated_songs_sorted == None or len(self.current_rated_songs_sorted)==0:
            return True        

        for var in vars:
            var:Variable
            if var.value in self.current_rated_songs_sorted:
                return False
        return True
    
    def __evaluate_all_diff(self,vars: tuple) -> bool:
        """ the recommendations must be differents """
        seen_values = set()
        for var in vars:
            var:Variable
            if var.value in seen_values:
                return False
            seen_values.add(var.value)
        return True
    
