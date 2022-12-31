import random as rnd
import pandas as pd
import time
from enum import Enum
Aimport similarity_measures as sim
import Agents_actions as act

class Listening_behavior(Enum):
    Indiferents = 1     # 40% # They would not lose much sleep if music ceased to exist, they are a predominant type of listeners of the whole population.
    Casuals = 2         # 32% # Music plays a welcome role, but other things are far more important
    Enthusiasts = 3     # 21% # Music is a key part of life but is also balanced by other interests.
    Savants = 4         #  7% # Everything in life seems to be tied up with music. Their musical knowledge is very extensive.

class Behavior_Distribution:
    def __init__(self, interact_dist:dict=None, explicit_dist:dict=None, change_internal_dist:dict=None):
        # if rate_dist is None:
        #     rate_dist = {'yes': 0.4, 'no': 0.6 }
        if interact_dist is None:
            interact_dist = {'empty'    : 0.30,
                            'search'    : 0.20,
                            'post'      : 0.15,
                            'read'      : 0.15}
        if change_internal_dist is None:
            change_internal_dist = {'yes': 0.3, 'no': 0.7}
        if explicit_dist is None:
            explicit_dist = {'none'         : 0.30,
                            'artists'       : 0.25,
                            'genres'        : 0.25,
                            'artist_genres' : 0.20}

        self.behavior = { 
            # 'rate_dist': rate_dist, 
                        'interact_dist': interact_dist, 
                        'change_internal_dist': change_internal_dist,
                        'explicit_dist': explicit_dist}
        
    def __iter__(self):
        return iter(self.behavior)
    def __getitem__(self, element):
        return self.behavior[element]

class Agent():
    """ Utility-based agents. Their utility function is listening songs similars to their preference """
    # def __init__(self, id: int, behavior_distributions:Behavior_Distribution, preference:dict, listenin_behavior:Listening_behavior):
    #     self.id = id
    #     self.preference = preference
    #     self.listenin_behavior = listenin_behavior
    #     if not behavior_distributions:
    #         self.behavior_dist = Behavior_Distribution()
    #     else: self.behavior_dist = behavior_distributions

    def register_in_system(self) -> act.Action:
        expl_dist:dict = self.behavior_dist['explicit_dist']
        # 'none', 'artists', 'genres', 'artist_genres'
        action_str = rnd.choices(list(expl_dist.keys()),list(expl_dist.values()))

        if action_str == 'none':
            return act.EmptyAction()
        elif action_str == 'artists':
            return act.RegisterAction(self.preference['artists'])
        elif action_str == 'genres':
            return act.RegisterAction(self.preference['genres'])
        elif action_str == 'artist_genres':
            return act.RegisterAction(self.preference['artists'], self.preference['genres'])
     
    def similarity_to_preference(self, song): 
        """ utility function """
        raise NotImplementedError()

    def received_recommendation(self, recommendation: list) -> act.Action:
        diff_function = lambda song, rate: (rate  - self.similarity_to_preference(song))**2
        rate_recommendation = self.__stochasctic_hill_climbing(100, diff_function, recommendation)
        return rate_recommendation

    def __stochasctic_hill_climbing(self, iterations:int, min_funct, songs_population):
        # pobl_size = len(sol_range)
        # initialization:
        rnd.seed(time.time())
        bounds = [0,5]
        rate_population = []
        for p in range(len(songs_population)):
            # initial solution
            rate = rnd.uniform(bounds[0], bounds[1])
            eval = min_funct(rate)
            rate_population.append([rate,eval])

        for it in range(iterations):
            # step
            for i in range(len(rate_population)):
                rnd.seed(time.time())
                r = rate_population[i]
                s = songs_population[i]
                candidate_rate = r[0] + rnd.uniform(-1,1)
                candidate_eval = min_funct(s,candidate_rate)
                if candidate_eval <= r[1]:
                    #improved solution
                    r[0] = candidate_rate
                    r[1] = candidate_eval
                # report progress
                print('> %s - iter: %d f(%s) = %.5f' % (str(s),it, r[0], r[1])) 

        return rate_population

    def do_action(self) -> act.Action:
        raise NotImplementedError()

class UniformAgent(Agent):
    """ Uniformily Random Preference User """
    def __init__(self, id: int,listenin_behavior = Listening_behavior.Casuals):
        self.id = id
        self.listenin_behavior = Listening_behavior.Casuals
        self.behavior_dist = Behavior_Distribution()
        self.preference = {}
     
    def similarity_to_preference(self, song): 
        """ utility function """
        sim = self.preference.get(song.id)
        if not sim:
            rnd.seed(time.time())
            self.preference[song.id] = rnd.uniform(0,1)
            sim = self.preference[song.id]
        return sim
    
    def do_action(self) -> act.Action:
        interact_dist:dict = self.behavior_dist['interact_dist']
        # 'empty', 'search', 'post', 'read'
        action = rnd.choices(list(interact_dist.keys()),list(interact_dist.values()))
        if action == 'empty':
            return act.EmptyAction()
        else: raise NotImplementedError()

class LooselyPreferenceAgent(Agent):
    """ Loosely Preference User """
    def __init__(self, id: int, preference_songs:list, listenin_behavior = Listening_behavior.Casuals):
        self.id = id
        self.listenin_behavior = listenin_behavior
        self.behavior_dist = Behavior_Distribution(interact_dist={'empty':0.30,'search':0.35,'post':0.10,'read':0.25}, change_internal_dist={'yes':0.8,'no':0.2})
        self.preference = {'songs': [], 'artists': [], 'genres': [] }
        for song in preference_songs:
            self.preference['songs'].append(song.id)
            self.preference['artists'].append(art for art in song.artists)
            self.preference['genres'].append(gen for gen in song.genres)
        # remove repeated ones
        self.preference['artists'] = list(set(self.preference['artists']))
        self.preference['genres'] = list(set(self.preference['genres']))
    
    def similarity_to_preference(self, song):        
        simil = sim.Similarity()
        gen_sim = simil.jaccard_similarity(self.preference['genres'], song.genres)
        art_sim = simil.jaccard_similarity(self.preference['artists'], song.artists)
        return max(gen_sim,art_sim)
 
    def received_recommendation(self, recommendation: list) -> act.Action:
        rate_recommendation = super().received_recommendation(recommendation)
        change_internal:dict = self.behavior_dist['change_internal_dist']
        # 'yes', 'no'
        add_to_preference = rnd.choices(list(change_internal.keys()),list(change_internal.values()))
        if add_to_preference == 'yes':
            for i in range(len(rate_recommendation)):
                rate=rate_recommendation[i]
                if rate[0] > 2.5:
                    self.preference['songs'].append(recommendation[i].id)
                    self.preference['artists'].append(art for art in recommendation[i].artists)
                    self.preference['genres'].append(gen for gen in recommendation[i].genres)
            # remove repeated ones
            self.preference['artists'] = list(set(self.preference['artists']))
            self.preference['genres'] = list(set(self.preference['genres']))

        return rate_recommendation
    
class StronglyPreferenceAgent(Agent):
    """ Strongly Preference User """
    def __init__(self, id: int, preference_songs:list, listenin_behavior = Listening_behavior.Casuals):
        self.id = id
        self.listenin_behavior = listenin_behavior
        self.behavior_dist = Behavior_Distribution(interact_dist={'empty':0.20,'search':0.30,'post':0.30,'read':0.20}, change_internal_dist={'yes':0.3,'no':0.7})
        self.preference = {'songs': [], 'artists': [], 'genres': [] }
        for song in preference_songs:
            self.preference['songs'].append(song.id)
            self.preference['artists'].append(art for art in song.artists)
            self.preference['genres'].append(gen for gen in song.genres)
        # remove repeated ones
        self.preference['artists'] = list(set(self.preference['artists']))
        self.preference['genres'] = list(set(self.preference['genres']))
    
    def similarity_to_preference(self, song):        
        simil = sim.Similarity()
        gen_sim = simil.jaccard_similarity(self.preference['genres'], song.genres)
        art_sim = simil.jaccard_similarity(self.preference['artists'], song.artists)
        return gen_sim*0.5 + art_sim*0.5
 
    def received_recommendation(self, recommendation: list) -> act.Action:
        rate_recommendation = super().received_recommendation(recommendation)
        change_internal:dict = self.behavior_dist['change_internal_dist']
        # 'yes', 'no'
        add_to_preference = rnd.choices(list(change_internal.keys()),list(change_internal.values()))
        if add_to_preference == 'yes':
            for i in range(len(rate_recommendation)):
                rate=rate_recommendation[i]
                if rate[0] >= 4.3:
                    self.preference['songs'].append(recommendation[i].id)
                    self.preference['artists'].append(art for art in recommendation[i].artists)
                    self.preference['genres'].append(gen for gen in recommendation[i].genres)
            # remove repeated ones
            self.preference['artists'] = list(set(self.preference['artists']))
            self.preference['genres'] = list(set(self.preference['genres']))

        return rate_recommendation
    
print(Listening_behavior.value(1))
 



