from math import*
from decimal import Decimal
import string
import numpy as np

class Similarity():
 
    """ Similarity measure functions """
 
    def euclidean_distance(self,x,y): 
        """ return euclidean distance between two lists """ 
        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 
    def manhattan_distance(self,x,y): 
        """ return manhattan distance between two lists """ 
        return sum(abs(a-b) for a,b in zip(x,y))
 
    def minkowski_distance(self,x,y,p_value): 
        """ return minkowski distance between two lists """ 
        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
           p_value)
 
    def nth_root(self,value, n_root): 
        """ returns the n_root of an value """ 
        root_value = 1/float(n_root)
        return round (Decimal(value) ** Decimal(root_value),3)
 
    # def cosine_similarity(self,x,y): 
    #     """ return cosine similarity between two lists """ 
    #     numerator = sum(a*b for a,b in zip(x,y))
    #     denominator = self.square_rooted(x)*self.square_rooted(y)
    #     return round(numerator/float(denominator),3)
    def cosine_similarity(self,x:np.array,y:np.array): 
        """ return cosine similarity between two arrays """ 
        numerator = np.dot(x, y)
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        denominator = x_norm * y_norm
        if numerator == 0 or denominator == 0:
            return 0
        else:
            return numerator/float(denominator)

    def square_rooted(self,x): 
        """ return 3 rounded square rooted value """ 
        return round(sqrt(sum([a*a for a in x])),3)
 
    def jaccard_similarity(self,x,y): 
        """ returns the jaccard similarity between two lists """ 
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

def _tf_x_idf(texts_list):
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
        pass
    terms_by_text, terms = process(texts_list)
    frq_m, t_dict = freq_matrix(terms, terms_by_text)
    idf_l = idf(frq_m,t_dict)
    tfXidf = tf_idf(frq_m, t_dict, terms, idf_l)
    return tfXidf

