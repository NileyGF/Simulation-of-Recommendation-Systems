from Songs_Modeling import Song
from Users_Modeling import User, Listening_behavior
from CSP import *
from Core import *
import Data
import util
import similarity_measures as sim
import Agents_actions as act
import time
import pickle
import pandas as pd
import random as rnd

dataframe = Data.read_songs_info('music_database_with_lists.csv')
usersframe = Data.read_users_songs_info('ratings.csv')

class Knowledge_based_recommender(Recommender):
    def __init__(self,songsFrame:pd.DataFrame,userFrame:pd.DataFrame):
        self.songsFrame = songsFrame
        self.extract_song_info()
        self.freq_matrix = util.vectorize_songs(self.songs_list)
        self.usersframe = userFrame
        self.profile_users()
        self.vector_by_user()
        self.user_conv = {}

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
    
    def extract_song_info(self):
        self.songs_list = Song.from_Dataframe(self.songsFrame)
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
        user = self.users_dict[self.user_conv[ag.id]]
        recommendation = self.csp_recommend(user,None,None)
        for i in range(len(recommendation)):
            recommendation[i] = recommendation[i].value
        ratings, changed = ag.received_recommendation(recommendation)
        for i in range(len(recommendation)):
            self.new_rate(user, recommendation[i], ratings[i])
        self.revector(user)
        return changed
        
    def csp_recommend(self, user:User,median_listen_count:int,median_rates:list):
        self.current_user_vector = user
        self.current_repetitions = user.songs_to_recommend
        self.current_pref_artists = user.prefered_artists
        self.current_pref_genders = user.prefered_genres
        self.current_rated_songs_sorted = user.top_rated_songs_ids(all=True)
        self.current_rated_songs_sorted = self.get_songs_from_ids(self.current_rated_songs_sorted.keys())
        song_vars = []
        for i in range(self.current_repetitions*4):
            song_vars.append(str("song_"+i))
        domain = self.songs_list.copy()
        song_vars = Variable.from_names_to_equal_domain(song_vars,domain)
        user_vars = {}
        user_vars['pref_arts'] = Variable([self.current_pref_artists],name='pref_arts')
        user_vars['pref_genrs'] = Variable([self.current_pref_genders],name='pref_genrs')
        user_vars['pref_songs'] = Variable([self.current_rated_songs_sorted],name='pref_songs')
        if user.__getattribute__('listening_behavior'):
            user_vars['listening_behavior'] = Variable([user.listening_behavior],name='listening_behavior')
        else:
            user_vars['listening_behavior'] = Variable([Listening_behavior.Casuals],name='listening_behavior')

        # constraint 1: if user is Indiferent, the songs must be similar to liked_songs(0.1), or a popular one (top 100)
        # constraint 1: if user is Casual, the songs must be similar to liked_songs(0.3), or a popular one (top 70)
        # constraint 1: if user is Enthusiast, the songs must be similar to liked_songs(0.5), or a popular one (top 50)   
        # constraint 1: if user is Savant, the songs must be similar to liked_songs(0.7), or a popular one (top 20)
        const1_vars = [user_vars["listening_behavior"]]
        for k in song_vars:
            const1_vars.append(song_vars[k])
        const1 = Constraint(const1_vars, self.__evaluate_similar_songs)
        # constraint 2: if user is Indiferent, the song must have a liked genre or artist (probability of bypassing this = 0.75)
        # constraint 2: if user is Casual, the song must have a liked genre or artist (probability of bypassing this = 0.6)
        # constraint 2: if user is Enthusiast, the songs must have a liked genre or artist (probability of bypassing this = 0.45) 
        # constraint 2: if user is Savant, the songs must have a liked genre or artist (probability of bypassing this = 0.3)
        const2_vars = [user_vars["listening_behavior"], user_vars["pref_arts"], user_vars["pref_genrs"]]
        for k in song_vars:
            const2_vars.append(song_vars[k])
        const2 = Constraint(const2_vars, self.__evaluate_artists_genres)
        # constraint 3: the recommendations cannot be in user.rated_songs
        const3_vars = [user_vars["pref_songs"]]
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
        problem_name = "map coloring problem"
        print("#" * 80)

        print("displaying performance results of solver: '", constraint_solver, sep='')
        overall_constraints_amount = str(len(recommendation_problem.get_constraints()))
        print("unsatisfied_constraints_amounts out of", overall_constraints_amount, "overall constraints:",
                unsatisfied_constraints_amounts)
        if hasattr(solution, "__len__"):
            print("solution lengths (number of assignment and unassignment actions):", histories_lengths)
        print("\nsolution: ")
        for var in solution:
            print(var)
        print("time results (seconds):", time_results)
        return solution

    def __evaluate_similar_songs(self,vars:tuple) -> bool:
        margen_dict = { Listening_behavior.Indiferents:    0.1,
                        Listening_behavior.Casuals:        0.3,
                        Listening_behavior.Enthusiasts:    0.5,
                        Listening_behavior.Savants:        0.7}
        list_behav = None
        recomm = []
        for v in vars:
            v:Variable
            if v.name == "listening_behavior":
                list_behav = v
            else:
                recomm.append(v)
        simil_list = []
        simil = sim.Similarity()
        for rec in recomm:
            rec:Variable
            song:Song = rec.value
            simil_list.append(simil.cosine_similarity(song.vector,self.current_user_vector))
        margen = margen_dict[list_behav.value]
        for s in simil_list:
            if simil_list[s] < margen:
                return False
        return True
    def __evaluate_artists_genres(self,vars:tuple) -> bool:
        bypass_dict = { Listening_behavior.Indiferents:    0.75,
                        Listening_behavior.Casuals:        0.6,
                        Listening_behavior.Enthusiasts:    0.45,
                        Listening_behavior.Savants:        0.3}
        list_behav = None
        pref_arts = None
        pref_genrs = None
        recomm = []
        for v in vars:
            v:Variable
            if v.name == "listening_behavior":
                list_behav = v
            elif v.name == "pref_arts":
                pref_arts = v
            elif v.name == "pref_genrs":
                pref_genrs = v
            else:
                recomm.append(v)
        bypass = bypass_dict[list_behav.value]
        simil_arts_list = []
        simil_genrs_list = []
        simil = sim.Similarity()
        for rec in recomm:
            rec:Variable
            song:Song = rec.value
            intersection_cardinality = len(set.intersection(*[set(song.artists), set(pref_arts.value)]))
            simil_arts_list.append(intersection_cardinality)
            intersection_cardinality = len(set.intersection(*[set(song.genres), set(pref_genrs.value)]))
            simil_genrs_list.append(intersection_cardinality)
        main_condition = True
        for s in recomm:
            main_condition = main_condition and ( simil_arts_list[s] >= 1 or simil_genrs_list[s] >=1)
        if main_condition:
            return True
        return rnd.choices([True,False],[bypass,1-bypass]) [0]

    def __evaluate_not_repeated(self,vars:tuple) -> bool:
        rated_songs = None
        recomm = []
        for v in vars:
            v:Variable
            if v.name == "pref_songs":
                rated_songs = v
            else:
                recomm.append(v)

        for var in vars:
            var:Variable
            if var.value in rated_songs:
                return False
        return True
    
    def __evaluate_all_diff(self,vars: tuple) -> bool:
        seen_values = set()
        for var in vars:
            var:Variable
            if var.value in seen_values:
                return False
            seen_values.add(var.val)
        return True
    
    def top_popular_songs(self, songs_list:list, top:int=20):
        sorted_by_listened = sorted(songs_list, key=lambda item: item.listen_count, reverse=True)
        
        while top < len(sorted_by_listened)-1 and sorted_by_listened[top].listen_count == sorted_by_listened[top+1].listen_count:
            top += 1
        self.top_pop_songs = sorted_by_listened[:top]

kn_b_recom = Knowledge_based_recommender()

# colors = ["red", "green", "blue"]
# names = {"wa", "nt", "q", "nsw", "v", "sa", "t"}

# name_to_variable_map = Variable.from_names_to_equal_domain(names, colors)
# const1 = Constraint([name_to_variable_map["sa"], name_to_variable_map["wa"]], all_diff_constraint_evaluator)
# const2 = Constraint([name_to_variable_map["sa"], name_to_variable_map["nt"]], all_diff_constraint_evaluator)
# const3 = Constraint([name_to_variable_map["sa"], name_to_variable_map["q"]], all_diff_constraint_evaluator)
# const4 = Constraint([name_to_variable_map["sa"], name_to_variable_map["nsw"]], all_diff_constraint_evaluator)
# const5 = Constraint([name_to_variable_map["sa"], name_to_variable_map["v"]], all_diff_constraint_evaluator)
# const6 = Constraint([name_to_variable_map["wa"], name_to_variable_map["nt"]], all_diff_constraint_evaluator)
# const7 = Constraint([name_to_variable_map["nt"], name_to_variable_map["q"]], all_diff_constraint_evaluator)
# const8 = Constraint([name_to_variable_map["q"], name_to_variable_map["nsw"]], all_diff_constraint_evaluator)
# const9 = Constraint([name_to_variable_map["nsw"], name_to_variable_map["v"]], all_diff_constraint_evaluator)
# const10 = Constraint([name_to_variable_map["t"]], always_satisfied)
# constraints = (const1, const2, const3, const4, const5, const6, const7, const8, const9, const10)
# map_coloring_problem = ConstraintProblem(constraints)

# start_time = time.process_time()
# solution = classic_heuristic_backtracking_search(map_coloring_problem, with_history=True)
# end_time = time.process_time()

# time_results = [round(end_time - start_time,4)]
# histories_lengths = []
# unsatisfied_constraints_amounts = []
# histories_lengths.append(len(solution))
# unsatisfied_constraints_amounts.append(len(map_coloring_problem.get_unsatisfied_constraints()))

# constraint_solver = "classic_heuristic_backtracking_search"
# problem_name = "map coloring problem"
# print("#" * 80)

# print("displaying performance results of solver: '", constraint_solver, "' with problem: '", problem_name, "'",
#         sep='')
# overall_constraints_amount = str(len(map_coloring_problem.get_constraints()))
# print("unsatisfied_constraints_amounts out of", overall_constraints_amount, "overall constraints:",
#         unsatisfied_constraints_amounts)
# if hasattr(solution, "__len__"):
#     print("solution lengths (number of assignment and unassignment actions):", histories_lengths)
# print("\nsolution: ")
# for var in solution:
#     print(var)
# print("time results (seconds):", time_results)
