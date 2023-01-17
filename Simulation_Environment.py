import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
# import statistics as sta
import time
import Agents as agent
import Agents_actions as act
import content_based
import knowledge_based
import Data
from Songs_Modeling import Song
from Core import *

class Profile_Generator:
    def __init__(self,songsFrame:pd.DataFrame,userFrame:pd.DataFrame):
        self.get_songs_dict(songsFrame)
        self.average_listened_songs(userFrame)
    
    def get_songs_dict(self,songsFrame:pd.DataFrame):
        self.__songs = {}
        self._song_list =[]
        for index, row in songsFrame.iterrows():
            self.__songs[row['song_id']] = {'title':row['title'], 'artists':row['artists'], 'genres':row['genres'], 'listen_count':row['listen_count']}
            self._song_list.append(Song(row['song_id'],row['title'],row['artists'], row['genres'], row['listen_count']))
        return self.__songs

    def average_listened_songs(self,userFrame:pd.DataFrame):
        song_user = userFrame.groupby('user_id')['song_id'].count()
        self.avg_songs = np.mean(song_user.values)
    
    def generate(self,num_users=99,pref_dist=[0.34,0.33,0.33],list_dist=[0.40,0.32,0.21,0.07]):
        sim_users = []
        rnd.seed(time.time())
        for i in range(num_users):
            user_id = i+1
            preference_str = rnd.choices(['random','loose','strong'],weights=pref_dist)[0]
            list_beh= rnd.choices([1,2,3,4],weights=list_dist) [0]
            listening_behavior = agent.Listening_behavior(list_beh)
            num_songs = int(self.avg_songs + rnd.randint(-4,5))
            if preference_str == 'random':
                sim_users.append(agent.UniformAgent(user_id,listening_behavior))
            elif preference_str == 'loose':
                preference = rnd.choices(self._song_list, k=num_songs)
                sim_users.append(agent.LooselyPreferenceAgent(user_id,preference,listening_behavior))
            elif preference_str == 'strong':
                preference = rnd.choices(self._song_list, k=num_songs)
                sim_users.append(agent.StronglyPreferenceAgent(user_id,preference,listening_behavior))
        return sim_users
      
class Music_store():
    def __init__(self, recommender:Recommender, close_time = 300):
        rnd.seed(time.time())
        self.num_arrivals = 0           # ammount of arrivals until self.time
        self.num_departures = 0         # ammount of departures until self.time
        self.num_post = 0
        self.num_read = 0
        self.users_in_store = 0         # ammount of users in the store
        self.time = 0                   # elapsed time  
        self.close_time = close_time    # no more arrivals are allowed
        t0 = rnd.gauss(5,0.5)
        if t0 < 0: t0 = 3
        self.next_arrival = int(t0 + 1)
        self.next_departure = float('inf')
        self.next_interact = float('inf')
        self.events = {'arrive':[],'departure':[],'empty':[],'post':[],'read':[]}      
        self.posts = []
        self.agents_changes = {} # agent.id : [songs added]
        self.recommender = recommender
            
    def user_arrival(self,user:agent.Agent):
        now = self.next_arrival
        lapse = rnd.gauss(5,0.5)
        if lapse < 0: lapse = 3
        self.next_arrival = int(now + lapse)
        self.num_arrivals +=1
        self.users_in_store += 1
        if self.users_in_store == 1:
            lapse1 = rnd.gauss(30,5)
            if lapse1 < 0: lapse1 = 15
            self.next_departure = int(now + lapse1)
            lapse2 = rnd.gauss(10,2)
            if lapse2 < 0: lapse2 = 7
            while lapse2 >= lapse1:
                lapse2 = rnd.gauss(10,2)
                if lapse2 < 0: lapse2 = 7
            self.next_interact = int(now + lapse2)
        self.events['arrive'].append((user.id, now))

        # efectuate the arrival and recommendation
        recom_user = self.recommender.new_user(user)
        changes = self.recommender.recommend(user)
        l = self.agents_changes.get(user.id)
        if l is None:
            self.agents_changes[user.id] = changes
        else: 
            for s in changes:
                self.agents_changes[user.id].append(s)
    
    def user_departure(self,user:agent.Agent):
        self.time = self.next_departure
        self.num_departures +=1
        self.users_in_store -=1
        if self.users_in_store == 0:
            self.next_departure = float('inf')
            self.next_interact = float('inf')
        else: 
            lapse = rnd.gauss(5,0.5)
            if lapse < 0: lapse = 3
            self.next_departure = int(self.time + lapse)
        self.events['departure'].append((user.id, self.time))
        
    def empty_interaction(self,user:agent.Agent,action:act.EmptyAction):
        now = self.next_interact
        lapse = rnd.gauss(10,2)
        if lapse < 0: lapse = 7
        self.next_interact = int(now + lapse)
        self.events['empty'].append((user.id, now))

    def user_post(self,user:agent.Agent,action:act.PostAction):
        # update store status
        now = self.next_interact
        lapse = rnd.gauss(10,2)
        if lapse < 0: lapse = 7
        self.next_interact = int(now + lapse)
        self.num_post += 1

        post = action.get_post(self.recommender.get_songs())
        self.events['post'].append((user.id, now, post))
        self.posts.append(post)
        
    def user_read(self,user:agent.Agent,action:act.ReadAction):
        # update store status
        now = self.next_interact
        lapse = rnd.gauss(10,2)
        if lapse < 0: lapse = 7
        self.next_interact = int(now + lapse)
        self.num_read += 1

        changes = action.read(self.posts)
        self.events['read'].append((user.id, now))
        l = self.agents_changes.get(user.id)
        if l is None:
            self.agents_changes[user.id] = changes
        else: 
            for s in changes:
                self.agents_changes[user.id].append(s)

class Model:
    def __init__(self, recommender_system:str):
        dataframe = Data.read_songs_info('music_database_with_lists.csv')
        usersframe = Data.read_users_songs_info('ratings.csv')
        
        recom_dict = {'content-based'   : content_based.ContentBasedRecommender(dataframe,usersframe),
                      'knowledge-based' : knowledge_based.Knowledge_based_recommender(dataframe,usersframe)}
        self.recom_str = recommender_system
        self.recommender = recom_dict.get(self.recom_str)
        self.store = Music_store(self.recommender)
        self.prof_gen = Profile_Generator(dataframe,usersframe)
    
    def simulate(self,repeat:int=30, duration:int=1440, num_users:int=100):
        self.agents_list = self.prof_gen.generate(num_users)
        self.store.close_time = duration
        self.changes_for_iter = {'uniform':[0]*repeat, 'loosely':[0]*repeat, 'strongly':[0]*repeat}
        for i in range(repeat):
            self.run()
            self.process_iteration(i,self.store.agents_changes)
            self.store = Music_store(self.recommender,duration)
        self.end_state(repeat)
    
    def run(self):  
        running = True
        agents_in_store = []
        while running:
            if self.store.time == self.store.next_arrival:
                # new arrival
                new_us_ind = rnd.randint(0, len(self.agents_list)-1)
                while self.agents_list[new_us_ind]in agents_in_store:
                    new_us_ind = rnd.randint(0, len(self.agents_list)-1)
                user:agent.Agent = self.agents_list[new_us_ind]
                agents_in_store.append(user)
                self.store.user_arrival(user)
            if self.store.time == self.store.next_interact:
                inter_us_ind = rnd.randint(0, len(agents_in_store)-1)
                user:agent.Agent = agents_in_store[inter_us_ind]
                # 'empty', 'post', 'read'       
                action = user.do_action()
                if type(action) is act.EmptyAction:
                    self.store.empty_interaction(user,action)
                elif type(action) is act.PostAction:
                    self.store.user_post(user,action)
                elif type(action) is act.ReadAction:
                    if len(self.store.posts) > 0:
                        self.store.user_read(user,action)
            if self.store.time == self.store.next_departure:
                # a user leaves the store
                leave_us_ind = rnd.randint(0, len(agents_in_store)-1)
                user:agent.Agent = agents_in_store[leave_us_ind]
                agents_in_store.remove(user)
                self.store.user_departure(user)
            self.store.time += 1
            running = (self.store.time <= self.store.close_time)

    def end_state(self,iterations):
        path_dict = {'content-based'    : 'content-based data',
                     'knowledge-based'  : 'knowledge-based data'}
        path = path_dict[self.recom_str] + '/'
        uniform = self.changes_for_iter['uniform']
        loosely = self.changes_for_iter['loosely']
        strongly = self.changes_for_iter['strongly']
        t = np.linspace(0,iterations,iterations)
        plt.figure(1)
        plt.scatter(t,uniform)
        plt.scatter(t,loosely)
        plt.scatter(t,strongly)
        plt.legend(['uniform random user', 'loosely preference user', 'strongly preference user'])
        plt.xlabel('Simulation runs')
        plt.ylabel('Amount of changes')
        plt.title('Amount of changes in user\'s preferences per run')
        title = path + 'results.png'
        plt.savefig(title)
        uniform_cum = []
        loosely_cum = []
        strongly_cum = []
        for i in range(iterations):
            if i == 0:
                uniform_cum.append(uniform[i])
                loosely_cum.append(loosely[i])
                strongly_cum.append(strongly[i])
            else:
                uniform_cum.append(uniform_cum[i-1] + uniform[i])
                loosely_cum.append(loosely_cum[i-1] + loosely[i])
                strongly_cum.append(strongly_cum[i-1] + strongly[i])
        plt.figure(2)
        plt.plot(t,uniform_cum)
        plt.plot(t,loosely_cum)
        plt.plot(t,strongly_cum)
        plt.legend(['uniform random user', 'loosely preference user', 'strongly preference user'])
        plt.xlabel('Simulation runs')
        plt.ylabel('Cumulative amount of changes')
        plt.title('Cumulative amount of changes in user\'s preferences per run')
        title = path + 'results cumulative.png'
        plt.savefig(title)
        
        print('Mean of uniform random users changes: ',np.mean(uniform))
        print('Median of uniform random users changes: ',np.median(uniform))
        print('Mean of loosely preference users changes: ',np.mean(loosely))
        print('Median of loosely preference users changes: ',np.median(loosely))
        print('Mean of strongly preference users changes: ',np.mean(strongly))
        print('Median of strongly preference users changes: ',np.median(strongly))
        
    def process_iteration(self,iter:int,changes_list):
        for id in changes_list:
            ag_index = id -1
            ag:agent.Agent = self.agents_list[ag_index]
            if type(ag) is agent.UniformAgent:
                self.changes_for_iter['uniform'][iter] += len(changes_list[id])
            elif type(ag) is agent.LooselyPreferenceAgent:
                self.changes_for_iter['loosely'][iter] += len(changes_list[id])
            elif type(ag) is agent.StronglyPreferenceAgent:
                self.changes_for_iter['strongly'][iter] += len(changes_list[id])
            else:
                pass

start = time.time()
model = Model('content-based')
model.simulate(repeat=30,duration=1440)
end = time.time()
print('running time:', round(end - start,4),'sec')

start = time.time()
model = Model('knowledge-based')
model.simulate(repeat=30,duration=700)
end = time.time()
print('running time:', round(end - start,4),'sec')