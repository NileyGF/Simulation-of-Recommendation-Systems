import random as rnd
class Action:
    """ rate, comment, no rate, change internal state, 
        rate in the system, or None"""
    pass
class EmptyAction(Action):
    pass
class RegisterAction(Action):
    def __init__(self,expl_artists:list=[],expl_genres:list=[]):
        self.expl_artists = expl_artists
        self.expl_genres = expl_genres
    def register_explicit(self,user): 
        user._explicit_prefered_artist(self.expl_artists)
        user._explicit_prefered_genres(self.expl_genres)
    
# class RateAction(Action):
#     pass
# class ChangeInternalAction(Action):
#     pass
class InteractAction(Action):
    pass
class PostAction(InteractAction):
    def __init__(self, agent, song_id):
        self.agent = agent
        self.song_id = song_id
    def get_post(self, songs_list):
        for s in songs_list:
            if s.id == self.song_id:
                song = s
                break
        return song

class ReadAction(InteractAction):
    def __init__(self, agent):
        self.agent = agent
    def read(self, posts:list):
        rates = []
        n = min(rnd.randint(1,3),len(posts))
        songs = list(set(rnd.choices(posts, k = n)))
        for i in range(len(songs)):
            r = rnd.randint(2,10) * 0.5
            rates.append(r)
        return self.agent.change_inter(songs,rates)
    