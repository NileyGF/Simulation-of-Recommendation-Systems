import random as rnd
import pandas as pd
import time
from enum import Enum
import similarity_measures as sim
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
            interact_dist = {'empty'    : 0.45,
                            'post'      : 0.30,
                            'read'      : 0.25}
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
        action_str = rnd.choices(list(expl_dist.keys()),list(expl_dist.values()))[0]

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
        bounds = [1,5]
        rate_population = []
        for p in range(len(songs_population)):
            # initial solution
            rate = rnd.uniform(bounds[0], bounds[1])
            eval = min_funct(songs_population[p],rate)
            rate_population.append([rate,eval])

        for it in range(iterations):
            # step
            for i in range(len(rate_population)):
                rnd.seed(time.time())
                r = rate_population[i]
                s = songs_population[i]
                candidate_rate = r[0] + rnd.uniform(-1,1)
                candidate_rate = max(0,candidate_rate)
                candidate_eval = min_funct(s,candidate_rate)
                if candidate_eval <= r[1]:
                    #improved solution
                    r[0] = candidate_rate
                    r[1] = candidate_eval
                # report progress
                # print('> %s - iter: %d f(%s) = %.5f' % (str(s),it, r[0], r[1])) 
        
        for p in range(len(rate_population)):
            r = rate_population[p][0]
            if r < 1: r = 0
            elif r > 1 and r < 1.5: r = 1.5
            elif r > 1.5 and r < 2: r = 2
            elif r > 2 and r < 2.5: r = 2.5
            elif r > 2.5 and r < 3: r = 3
            elif r > 3 and r < 3.5: r = 3.5
            elif r > 3.5 and r < 4: r = 4
            elif r > 4 and r < 4.5: r = 4.5
            elif r > 4.5 and r < 5: r = 6
            rate_population[p][0] = r

        return rate_population

    def do_action(self) -> act.Action:
        raise NotImplementedError()

    def change_inter(self,songs,rates:list):
        raise NotImplementedError()


class UniformAgent(Agent):
    """ Uniformily Random Preference User """
    def __init__(self, id: int,listenin_behavior = Listening_behavior.Casuals):
        self.id = id
        self.listenin_behavior = Listening_behavior.Casuals
        self.behavior_dist = Behavior_Distribution()
        self.preference = {}
   
    def register_in_system(self) -> act.Action:
        return act.EmptyAction()

    def similarity_to_preference(self, song): 
        """ utility function """
        sim = self.preference.get(song.id)
        if not sim:
            rnd.seed(time.time())
            self.preference[song.id] = rnd.uniform(0,1)
            sim = self.preference[song.id]
        return sim
    
    def received_recommendation(self, recommendation: list) -> act.Action:
        rate_recommendation = super().received_recommendation(recommendation)
        rates = [r[0] for r in rate_recommendation]
        changed = self.change_inter(recommendation,rates)
        return rates, changed

    def change_inter(self, songs, rates: list):
        change_internal:dict = self.behavior_dist['change_internal_dist']
        # 'yes', 'no'
        add_to_preference = rnd.choices(list(change_internal.keys()),list(change_internal.values()))[0]
        changed = []
        if add_to_preference == 'yes':
            for i in range(len(rates)):
                if rates[i] > 0:
                    self.preference[songs[i].id] = rnd.uniform(0,1)
                    changed.append(songs[i])
        return changed

    def do_action(self) -> act.Action:
        interact_dist:dict = self.behavior_dist['interact_dist']
        # 'empty', 'post', 'read'
        action = rnd.choices(list(interact_dist.keys()),list(interact_dist.values()))[0]
        if action == 'empty':
            return act.EmptyAction()
        elif action == 'post':
            s_index = rnd.randint(0, len(self.preference)-1)
            s_id = list(self.preference.keys()) [s_index]
            return act.PostAction(self,s_id)
        elif action == 'read':
            return act.ReadAction(self)

class LooselyPreferenceAgent(Agent):
    """ Loosely Preference User """
    def __init__(self, id: int, preference_songs:list, listenin_behavior = Listening_behavior.Casuals):
        self.id = id
        self.listenin_behavior = listenin_behavior
        self.behavior_dist = Behavior_Distribution(interact_dist={'empty':0.50,'post':0.15,'read':0.35}, change_internal_dist={'yes':0.8,'no':0.2})
        self.preference = {'songs': [], 'artists': [], 'genres': [] }
        for song in preference_songs:
            self.preference['songs'].append(song.id)
            for art in song.artists:
                self.preference['artists'].append(art)
            for gen in song.genres:
                self.preference['genres'].append(gen)
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
        rates = [r[0] for r in rate_recommendation]
        changed = self.change_inter(recommendation,rates)
        return rates, changed
    
    def change_inter(self,songs,rates:list):
        change_internal:dict = self.behavior_dist['change_internal_dist']
        # 'yes', 'no'
        add_to_preference = rnd.choices(list(change_internal.keys()),list(change_internal.values()))[0]
        changed = []
        if add_to_preference == 'yes':
            for i in range(len(rates)):
                if rates[i] >= 2.5:
                    self.preference['songs'].append(songs[i].id)
                    for art in songs[i].artists:
                        self.preference['artists'].append(art)
                    for gen in songs[i].genres:
                        self.preference['genres'].append(gen)
                    changed.append(songs[i])
            # remove repeated ones
            self.preference['artists'] = list(set(self.preference['artists']))
            self.preference['genres'] = list(set(self.preference['genres']))
        return changed

    def do_action(self) -> act.Action:
        interact_dist:dict = self.behavior_dist['interact_dist']
        # 'empty', 'post', 'read'
        action = rnd.choices(list(interact_dist.keys()),list(interact_dist.values()))[0]
        if action == 'empty':
            return act.EmptyAction()
        elif action == 'post':
            s_id = rnd.choice(self.preference['songs'])
            return act.PostAction(self,s_id)
        elif action == 'read':
            return act.ReadAction(self)

class StronglyPreferenceAgent(Agent):
    """ Strongly Preference User """
    def __init__(self, id: int, preference_songs:list, listenin_behavior = Listening_behavior.Casuals):
        self.id = id
        self.listenin_behavior = listenin_behavior
        self.behavior_dist = Behavior_Distribution(interact_dist={'empty':0.40,'post':0.40,'read':0.20}, change_internal_dist={'yes':0.3,'no':0.7})
        self.preference = {'songs': [], 'artists': [], 'genres': [] }
        for song in preference_songs:
            self.preference['songs'].append(song.id)
            for art in song.artists:
                self.preference['artists'].append(art)
            for gen in song.genres:
                self.preference['genres'].append(gen)
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
        rates = [r[0] for r in rate_recommendation]
        changed = self.change_inter(recommendation,rates)
        return rates, changed
    
    def change_inter(self,songs,rates:list):
        change_internal:dict = self.behavior_dist['change_internal_dist']
        # 'yes', 'no'
        add_to_preference = rnd.choices(list(change_internal.keys()),list(change_internal.values()))[0]
        changed = []
        if add_to_preference == 'yes':
            for i in range(len(rates)):
                if rates[i] >= 4.0:
                    self.preference['songs'].append(songs[i].id)
                    for art in songs[i].artists:
                        self.preference['artists'].append(art)
                    for gen in songs[i].genres:
                        self.preference['genres'].append(gen)
                    changed.append(songs[i])
            # remove repeated ones
            self.preference['artists'] = list(set(self.preference['artists']))
            self.preference['genres'] = list(set(self.preference['genres']))
        return changed

    def do_action(self) -> act.Action:
        interact_dist:dict = self.behavior_dist['interact_dist']
        # 'empty', 'post', 'read'
        action = rnd.choices(list(interact_dist.keys()),list(interact_dist.values()))[0]
        if action == 'empty':
            return act.EmptyAction()
        elif action == 'post':
            s_id = rnd.choice(self.preference['songs'])
            return act.PostAction(self,s_id)
        elif action == 'read':
            return act.ReadAction(self)

# print(Listening_behavior(1))
 



