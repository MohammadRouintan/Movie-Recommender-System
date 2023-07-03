import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MinMaxScaler
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

class Model:

    def __init__(self, model_name):
        self.model_name = ''
        self.model_data = ''
        if model_name == 'Movie':
            self.model_name = model_name
            self.model_data = pd.read_csv('data/MovieBasedRecommender.csv')
        elif model_name == 'User':
            self.model_name = model_name
            self.model_data = pd.read_csv('data/UserBasedRecommender.csv')


    def calculate_tfidf(data):
        tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data)
        return tfidf_matrix


    def combine_recommendation(title, movie_indices):
        idx = movies_list[movies_list['title'] == title].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[int(idx)], tfidf_matrix)
        similarity_scores = list(enumerate(cosine_sim[0]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:501]
        movie_indices_combine = [i[0] if i[0] in movie_indices else None for i in similarity_scores]

        names = set()
        for i in movie_indices_combine:
            if i != None:
                names.add(titles.iloc[i])
        return names, set(movie_indices_combine)

    def recommender(title):
        idx = movies_list[movies_list['title'] == title].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[int(idx)], tfidf_matrix)
        similarity_scores = list(enumerate(cosine_sim[0]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:501]
        movie_indices = [i[0] for i in similarity_scores]

        posters = []
        for i in movie_indices:
            posters.append(api_poster(ids.iloc[i]))
        names = []
        for i in movie_indices:
            names.append(titles.iloc[i])
        return names, posters, movie_indices