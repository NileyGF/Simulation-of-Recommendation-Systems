import pandas as pd
import numpy as np
import random as rnd
import statistics as sta
from heapq import heappush, heappop
from itertools import count
import time
import Agents as agent
import Agents_actions as act
import content_based
import Data
from Songs_Modeling import Song

class Profile_Generator:
    def __init__(self,songsFrame:pd.DataFrame,userFrame:pd.DataFrame):
        self.get_songs_dict(songsFrame)
        self.average_listened_songs(userFrame)
    
    def get_songs_dict(self,songsFrame:pd.DataFrame):
        self.__songs = {}
        self._song_list =[]
        for index, row in songsFrame.iterrows():
            self.__songs[row['song_id']] = {'title':row['title'], 'artists':row['artists'], 'genres':row['genres'], 'listen_count':row['listen_count']}
            self._song_list.append(Song(row['song_id'],row['title'],row['artists'], row['genres']))
        return self.__songs

    def average_listened_songs(self,userFrame:pd.DataFrame):
        song_user = userFrame.groupby('user_id')['song_id'].count()
        self.avg_songs = np.mean(song_user.values)
    
    def generate(self,num_users=99,pref_dist=[0.33,0.67,1],list_dist=[0.40,0.32,0.21,0.07]):
        sim_users = []
        rnd.seed(time.time())
        for i in range(num_users):
            user_id = i+1
            preference_str = rnd.choices(['random','loose','strong'],cum_weights=pref_dist)
            list_beh= rnd.choices([1,2,3,4],weights=list_dist)
            listening_behavior = agent.Listening_behavior(list_beh)
            num_songs = self.avg_songs + rnd.randint(-4,5) #[7,10,15,20] [list_beh-1]
            if preference_str == 'random':
                sim_users.append(agent.UniformAgent(user_id,listening_behavior))
            elif preference_str == 'loose':
                preference = rnd.sample(self.__songs, num_songs)
                sim_users.append(agent.LooselyPreferenceAgent(user_id,preference,listening_behavior))
            elif preference_str == 'strong':
                preference = rnd.sample(self.__songs, num_songs)
                sim_users.append(agent.LooselyPreferenceAgent(user_id,preference,listening_behavior))
        return sim_users

PENDING = object()
class Event:
    """An event that may happen at some point in time.
    An event
    - may happen (:attr:`triggered` is ``False``),
    - is going to happen (:attr:`triggered` is ``True``) or
    - has happened (:attr:`processed` is ``True``)."""
    def __init__(self, env):

        self.env = env          # The class:Environment the event lives in.
        self.callbacks = []     # List of functions that are called when the event is processed.
        self._value = PENDING
    @property
    def triggered(self):
        """Becomes True if the event has been triggered and its callbacks are about to be invoked."""
        return self._value is not PENDING
    @property
    def processed(self):
        """Becomes True if the event has been processed (e.g., its callbacks have been invoked)."""
        return self.callbacks is None
    @property
    def value(self):
        """The value of the event if it is available.
        The value is available when the event has been triggered.
        Raises :exc:`AttributeError` if the value is not yet available.
        """
        if self._value is PENDING:
            raise AttributeError('Value of %s is not yet available' % self)
        return self._value
    def trigger(self, value=None):
        """Trigger the event with the value provided.
        Raises :exc:RuntimeError if this event has already been triggerd.
        """
        if self._value is not PENDING:
            raise RuntimeError('%s has already been triggered' % self)

        self._ok = True
        self._value = value
        self.env.schedule(self)
        return self

class Enviroment:
    def __init__(self, initial_time=0):
        self._now = initial_time
        self._queue = []  # The list of all currently scheduled events.
        self._eid = count()  # Counter for event IDs
        self._active_proc = None
    @property
    def now(self):
        """The current simulation time."""
        return self._now
    def schedule(self, event, delay=0):
        """Schedule an *event* with a given *delay*."""
        heappush(self._queue, (self._now + delay, next(self._eid), event))
    def peek(self):
        """Get the time of the next scheduled event. 
        Return infinity if there is no further event."""
        try:
            return self._queue[0][0]
        except IndexError:
            return float('inf')
    def run(self, until=None):
        """Executes :meth:`step()` until the given criterion *until* is met.
        - If it is ``None`` (which is the default), this method will return
          when there are no further events to be processed.
        - If it is a number, the method will continue stepping
          until the environment's time reaches *until*.
        """
        at = float(until)

        if at <= self.now:
            raise ValueError('until(=%s) should be > the current simulation time.' % at)

        # Schedule the event before all regular timeouts.
        until = Event(self)
        until._ok = True
        until._value = None
        self.schedule(until, at - self.now)            

        try:
            while True:
                self.step()
        except: 
            if until is not None:
                assert not until.triggered
                raise RuntimeError('No scheduled events left but "until" event was not triggered: %s' % until)
    def step(self):
        """Process the next event. Raise an RuntimeError if no further events are available. """
        try:
            self._now, _, event = heappop(self._queue)
        except IndexError:
            raise RuntimeError('No scheduled events left')

        # Process callbacks of the event. Set the events callbacks to None
        # immediately to prevent concurrent modifications.
        callbacks, event.callbacks = event.callbacks, None
        for callback in callbacks:
            callback(event)
    # def timeout(self,event:Event,delay,value=None):
    #     if delay < 0:
    #         raise ValueError('Negative delay %s' % delay)
    #     event.callbacks = []
    #     event._value = value
    #     event._delay = delay
    #     event._ok = True
    #     self.schedule(event, delay)
    # def initialize(self,event:Event,process):
    #     event.callbacks = [process._resume]
    #     event._value = None
    #     event._ok = True
    #     self.schedule(event)
    # def process(self,event:Event,generator):
    #     event.callbacks = []
    #     event._value = PENDING
    #     event._generator = generator
    #     event._target = self.initialize(self,event)
        

class Music_store():
    def __init__(self, recommender:content_based.ContentBasedRecommender, close_time = 300):
        rnd.seed(30)
        self.num_arrivals = 0           # ammount of arrivals until self.time
        self.num_departures = 0         # ammount of departures until self.time
        self.num_post = 0
        self.num_read = 0
        self.users_in_store = 0         # ammount of users in the store
        self.time = 0                   # elapsed time  
        self.close_time = close_time    # no more arrivals are allowed
        t0 = rnd.expovariate(1)
        self.next_arrival = t0
        self.next_departure = float('inf')
        self.next_interact = float('inf')
        self.events = {'arrive':[],'departure':[],'empty':[],'post':[],'read':[]}      
        self.posts = []
        self.recommender = recommender
            
    def user_arrival(self,user:agent.Agent):
        # rnd.seed(time.time())
        # self.next_arrival = min(self.next_arrival,self.next_departure,self.close_time)
        # if not (self.next_arrival <= self.next_departure and self.next_arrival <= self.close_time):
        #     print("Nooooo")
        #     return 
        # update store status
        now = self.next_arrival
        lapse = rnd.expovariate(1) + 1
        self.next_arrival = int(now + lapse)
        self.num_arrivals +=1
        self.users_in_store += 1
        if self.users_in_store == 1:
            lapse1 = rnd.expovariate(1) + 1
            self.next_departure = int(now + lapse)
            lapse2 = rnd.expovariate(1)
            while lapse2 >= lapse1:
                lapse2 = rnd.expovariate(1) + 1
            self.next_interact = int(now + lapse)
        self.events['arrive'].append((user.id, now))

        # efectuate the arrival and recommendation
        recom_user = self.recommender.new_user(user)
        self.recommender.recommend(recom_user)
    
    def user_departure(self,user:agent.Agent):
        # rnd.seed(time.time())
        # if not (self.next_departure <= self.next_arrival and self.next_departure <= self.close_time):
        #     print("Nooooo")
        #     return 
        self.time = self.next_departure
        self.num_departures +=1
        self.users_in_store -=1
        if self.users_in_store == 0:
            self.next_departure = float('inf')
            self.next_interact = float('inf')
        else: 
            lapse = rnd.expovariate(1) + 1
            self.next_departure = self.time + lapse
        self.events['departure'].append((user.id, self.time))
        
    def empty_interaction(self,user:agent.Agent,action:act.EmptyAction):
        # if not (self.next_interact <= self.next_departure and self.next_interact <= self.next_arrival):
        #     print("Nooooo")
        #     return 
        # update store status
        now = self.next_interact
        lapse = rnd.expovariate(3) + 1
        self.next_interact = int(now + lapse)
        self.events['empty'].append((user.id, now))

    def user_post(self,user:agent.Agent,action:act.PostAction):
        # update store status
        now = self.next_interact
        lapse = rnd.expovariate(3) + 1
        self.next_interact = int(now + lapse)
        self.num_post += 1

        post = action.get_post(self.recommender.content_simil.songs_list)
        self.events['post'].append((user.id, now, post))
        self.posts.append(post)
        
    def user_read(self,user:agent.Agent,action:act.ReadAction):
        # update store status
        now = self.next_interact
        lapse = rnd.expovariate(3) + 1
        self.next_interact = int(now + lapse)
        self.num_read += 1

        action.read(self.posts)
        self.events['read'].append((user.id, now))

class Model:
    def __init__(self, recommender_system):
        dataframe = Data.read_songs_info('music_database_with_lists.csv')
        usersframe = Data.read_users_songs_info('ratings.csv')
        self.recommender = content_based.ContentBasedRecommender(dataframe,usersframe)
        self.store = Music_store(self.recommender)
        self.prof_gen = Profile_Generator(dataframe,usersframe)
    
    def simulate(self, duration:int=300, num_users:int=99):
        self.agents_list = self.prof_gen.generate(num_users)
        self.store.close_time = duration
        self.run()
    
    def run(self):  
        running = True
        agents_in_store = []
        while running:
            if self.store.time == self.store.next_arrival:
                # new arrival
                new_us_ind = rnd.randint(0, len(self.agents_list)-1)
                while self.agents_list[new_us_ind] in agents_in_store:
                    new_us_ind = rnd.randint(0, len(self.agents_list)-1)
                user:agent.Agent = self.agents_list[new_us_ind]
                agents_in_store.append(user)
                self.store.user_arrival(user)
            if self.store.time == self.store.next_interact:
                inter_us_ind = rnd.randint(0, len(agents_in_store)-1)
                user:agent.Agent = agents_in_store[inter_us_ind]
                # 'empty', 'post', 'read'       
                action = user.do_action()
                if action is act.EmptyAction:
                    self.store.empty_interaction(user,action)
                elif action is act.PostAction:
                    self.store.user_post(user,action)
                elif action is act.ReadAction:
                    self.store.user_read(user,action)
            if self.store.time == self.store.next_departure:
                # a user leaves the store
                leave_us_ind = rnd.randint(0, len(agents_in_store)-1)
                user:agent.Agent = agents_in_store[leave_us_ind]
                agents_in_store.remove(user)
                self.store.user_departure(user)
            self.store.time += 1
            running = (self.store.time <= self.store.close_time)

model = Model('content_based')
model.simulate()