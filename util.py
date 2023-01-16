import string
import numpy as np

def get_Id_Ind_dict(song_list):
    id_ind = {}
    for i in range(len(song_list)):
        id_ind[song_list[i].id] = i
    return id_ind
class Util:
    info_by_song = None  # [terms_by_title, artists_by_song, genres_by_song]
    sorted_vocab = None  # [title_vocabulary, unique_artists, unique_genres]

def vectorize_songs(song_list:list):
    if Util.info_by_song == None or Util.sorted_vocab == None:
        terms_by_title, artists_by_song, genres_by_song, titles_vocabulary, artists, genres = process_song_list(song_list)
    else:
        terms_by_title = Util.info_by_song[0]
        artists_by_song = Util.info_by_song[1]
        genres_by_song = Util.info_by_song[2]
        titles_vocabulary = Util.sorted_vocab[0]
        artists = Util.sorted_vocab[1]
        genres = Util.sorted_vocab[2]

    shape = (len(titles_vocabulary) + len(artists) + len(genres), len(song_list))
    freq_matrix = np.ndarray(shape, dtype=int)

    for j in range(shape[1]):
        for i in range(len(titles_vocabulary)):
            freq = terms_by_title[j].count(titles_vocabulary[i])
            freq_matrix[i,j] = freq
    ind = len(titles_vocabulary)

    for j in range(len(artists_by_song)):  # len(artists_by_song) == shape[1]
        for art in artists_by_song[j]:
            i = artists.index(art)
            freq_matrix[ind+i, j] = 1
    ind = len(titles_vocabulary) + len(artists)
    
    for j in range(len(genres_by_song)):   # len(genres_by_song) == shape[1]
        for gen in genres_by_song[j]:
            i = genres.index(gen)
            freq_matrix[ind+i, j] = 1
    
    for s in range(len(song_list)):
        song_list[s].vector = freq_matrix[:,s]
        
    return freq_matrix

def tokenize_text(text:str):
    text = text.lower()
    # remove numbers
    text = text.translate(str.maketrans('', '', string.digits))
    # remove punctuation (replace them with ' ')
    text = text.translate(str.maketrans("!\"'()*+,-./:;<=>?@[]\\^_`{|}~", ' '*len("!\"'()*+,-./:;<=>?@[]\\^_`{|}~")))
    # generate tokens
    tokens = text.split()
    # remove stopwords and lemmatize
    return tokens
def process(texts_list:list):
    # list of all indexed terms
    indexed_terms = []
    terms_by_text = []
    for i in range(len(texts_list)):
        tokens = tokenize_text(texts_list[i])
        # assign the terms of the text i
        terms_by_text.append(tokens)    
        for x in tokens:         
            # update the indexed terms list       
            indexed_terms.append(x)        
    
    unique_words = list(set(indexed_terms))  # remove repeated terms by using set()
    unique_words.sort()                      # sort to have a unique order when indexing
    return terms_by_text, unique_words
def process_song_list(song_list:list):
    # list of all indexed terms
    title_vocabulary = []
    artists = []
    genres = []
    terms_by_title = []
    artists_by_song = []
    genres_by_song = []
    for i in range(len(song_list)):
        s = song_list[i]
        # process title
        title_tokens = tokenize_text(s.title)
        terms_by_title.append(title_tokens)    
        for x in title_tokens:              
            title_vocabulary.append(x) 
        # process artists
        artists_by_song.append(s.artists)
        for x in s.artists:
            artists.append(x)
        # process genres     
        genres_by_song.append(s.genres)
        for x in s.genres:
            genres.append(x)  
    
    title_vocabulary = list(set(title_vocabulary))  # remove repeated terms by using set()
    title_vocabulary.sort()                         # sort to have a unique order when indexing
    unique_artists = list(set(artists))             # remove repeated terms by using set()
    unique_artists.sort()                           # sort to have a unique order when indexing
    unique_genres = list(set(genres))               # remove repeated terms by using set()
    unique_genres.sort()                            # sort to have a unique order when indexing
    
    Util.info_by_song = [terms_by_title, artists_by_song, genres_by_song]
    Util.sorted_vocab = [title_vocabulary, unique_artists, unique_genres]

    return terms_by_title, artists_by_song, genres_by_song, title_vocabulary, unique_artists, unique_genres

def freq_matrix(terms:list, terms_by_text:list):
    text_by_term_dict = {t: [] for t in terms}
    for i in range(len(terms_by_text)):
        no_repeat = set(terms_by_text[i])
        #for every unique term in the text, add to dict[term] the text index
        for t in no_repeat:
            text_by_term_dict[t].append(i)
    
    freq_matrix = np.ndarray((len(terms),len(terms_by_text)), dtype=int)
    # fill the non-zero positions of the frequency matrix 
    for i in range(len(terms)):
        for k in range(len(text_by_term_dict[terms[i]])):  
            # only count the frequency of a term for the documents it's in  
            ind = text_by_term_dict[terms[i]] [k]
            freq = terms_by_text[ind].count(terms[i])
            freq_matrix[i,ind] = freq
    return freq_matrix, text_by_term_dict
def idf(freq_matrix:np.ndarray, texts_by_term_dict:dict):
    """ idf[i] = log(total_texts/ number of docs where is the term i) """
    total_texts = freq_matrix.shape[1]
    idf = []
    for term in texts_by_term_dict:
        idf.append(np.log(total_texts / len(texts_by_term_dict[term])))
    return idf
def tf_idf(freq_matrix:np.ndarray,t_dict:dict,terms:list,idf:list):
    """ tf[i,j] = freq[i,j] / max freq[j]
        tfxidf[i,j] = tf[i,j] * idf[i]
    """
    max_freq = freq_matrix.max(axis=0,keepdims=True)
    tf_x_idf = np.ndarray(freq_matrix.shape, dtype=float)
    # fill the non-zero positions       
    for i in range(len(terms)):
        for k in range(len(t_dict[terms[i]])):
            ind = t_dict[terms[i]] [k]                 
            tf_i_j = freq_matrix[i,ind] / max_freq[0,ind]
            tf_x_idf[i,ind] = tf_i_j * idf[i]

    return tf_x_idf

def tf_x_idf(texts_list):
    terms_by_text, terms = process(texts_list)
    frq_m, t_dict = freq_matrix(terms, terms_by_text)
    idf_l = idf(frq_m,t_dict)
    tfXidf = tf_idf(frq_m, t_dict, terms, idf_l)
    return tfXidf
