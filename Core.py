import pandas as pd
import Agents as agent
from Users_Modeling import User
class Recommender:
    def __init__(self,songsFrame:pd.DataFrame,userFrame:pd.DataFrame):
        pass
    def new_user(self,agent:agent.Agent):
        pass

    def _print_message(self, user:User, recom_song):
        rec_items = len(recom_song)
        
        print(f'The {rec_items} recommended songs for {user} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][0]} with {round(recom_song[i][1], 3)} similarity score") 
            print("--------------------")
    
    def recommend(self,ag:agent.Agent):
        pass
    def get_songs(self):
        pass