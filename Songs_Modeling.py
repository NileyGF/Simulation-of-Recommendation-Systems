import pandas as pd
import numpy as np
import similarity_measures as sim
import util

class Song:
    def __init__(self, id:int, title:str, artists:list, genres:list, listen_count:int):
        self.id = id
        self.title = title
        self.artists = artists
        self.genres = genres
        self.listen_count = listen_count

    def __str__(self) -> str:
        string = self.title + ' - '
        for ar in self.artists:
            string += ar + ', '
        string = string[:len(string)-2]
        return string
    def __repr__(self) -> str:
        return self.__str__()

    def from_Dataframe(DataFrame:pd.DataFrame):
        ret = []
        for index, row in DataFrame.iterrows():
            ret.append(Song(row['song_id'],row['title'], row['artists'], row['genres'], row['listen_count']))
        return ret
    
    def song_similarity_heuristic(self,other, freq_matrix):
        """ Define the similarity among songs as the average of title similarity, artists similarity and genres similarity
            The title similarity is computed as a NLP problem, returning a cosine similarity, or 1-euclidean
            The artists and genres similarity is computed with jaccard similarity
        """
        simil = sim.Similarity()
        title_sim = simil.cosine_similarity(freq_matrix[:,self.id-1],freq_matrix[:,other.id-1])
        artists_sim = simil.jaccard_similarity(self.artists, other.artists)
        genres_sim = simil.jaccard_similarity(self.genres, other.genres)
        
        return 0.2 * title_sim + 0.4 * artists_sim + 0.4 * genres_sim

    def top_similar_songs(self, songs_list, top=50, freq_matrix=None):
        if freq_matrix is None:
            freq_matrix = util.vectorize_songs(songs_list)
        heurist = {}
        for s in songs_list:
            if s.id == self.id:
                continue
            heurist [s.id] = self.song_similarity_heuristic(s, freq_matrix)

        heurist = dict(sorted(heurist.items(), key=lambda item: item[1], reverse=True))
        ids = list(heurist.keys())
        self.similarities = []
        count = 0
        while count < top:
            self.similarities.append( (songs_list[ids[count]-1] , heurist[ids[count]]) )
            count += 1
        return self.similarities
           
    def _cosine_list(freq_matrix:np.ndarray,vector):
        cosine_sim = []
        v_norm = np.linalg.norm(vector)
        
        for t in range(freq_matrix.shape[1]):
            t_norm = np.linalg.norm(freq_matrix[:,t])
            norm_prod = v_norm * t_norm
            t_x_v = np.dot(freq_matrix[:,t],vector)

            if t_x_v == 0 or norm_prod == 0:
                cosine_sim.append(0)
            else:
                cosine_sim.append( t_x_v / norm_prod )
        return cosine_sim

    def title_corr_matrix(songs_list:list):
        freq_matrix = util.vectorize_songs(songs_list)
        corr = np.ndarray((len(songs_list),len(songs_list)), dtype=float)
        for t in range(freq_matrix.shape[1]):
            corr[:,t] = Song._cosine_list(freq_matrix,freq_matrix[:,t])
        return corr
    
