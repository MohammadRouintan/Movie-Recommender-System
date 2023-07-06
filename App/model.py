import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import requests
import joblib
import concurrent.futures

class Model:

    def __init__(self, model_name):
        self.model_name = ''
        self.model_data = ''
        if model_name == 'Movie':
            self.model_name = model_name
            self.model_data = pd.read_csv('data/MovieBasedRecommender.csv')
        elif model_name == 'User':
            self.model_name = model_name
            self.model_data = pd.read_csv('data/MovieBasedRecommender.csv')
            self.model_svd = joblib.load('model/SVD.joblib')
            self.model_id = pd.read_csv('data/movie_id.csv')
            self.model_ratings = pd.read_csv('data/ratings_small.csv')
        elif model_name == 'User & Movie':
            self.model_name = model_name
            self.model_data = pd.read_csv('data/MovieBasedRecommender.csv')
            self.model_svd = joblib.load('model/SVD.joblib')
            self.model_id = pd.read_csv('data/movie_id.csv')
            self.model_ratings = pd.read_csv('data/ratings_small.csv')

        self.model_feature = self.model_data['model_feature'].fillna('')
        self.tfidf_matrix = self.calculate_tfidf(self.model_feature)

    def api_poster(self, moive_id):
        url = "https://api.themoviedb.org/3/movie/{}?api_key=6b8ac0d80d854198821514621e36ae32&language=en-US".format(moive_id)
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        response = response.json()

        full_path = 'data/No-Image-Placeholder.png'
        if 'poster_path' in response:
            poster_path = response['poster_path']
            if not (poster_path == np.nan or poster_path == None): 
                full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

        return full_path

    def calculate_tfidf(self, data):
        tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data)
        return tfidf_matrix


    def combine_recommendation(self, title, movie_indices):
        data = self.model_data.reset_index()
        idx = data[data['title'] == title].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix[int(idx)], self.tfidf_matrix)
        similarity_scores = list(enumerate(cosine_sim[0]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:501]
        movie_indices_combine = [i[0] if i[0] in movie_indices else None for i in similarity_scores]

        names = set()
        for i in movie_indices_combine:
            if i != None:
                names.add(self.model_data['title'].iloc[i])
        return names, set(movie_indices_combine)

    def hybrid_svd(self, userId, title):
        data = self.model_data.reset_index()
        idx = data[data['title'] == title].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix[int(idx)], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:20]
        movie_indices = [i[0] for i in sim_scores]

        movies = self.model_data.iloc[movie_indices][['title', 'id']]
        tempLinks = self.model_id[self.model_id['tmdbId'].isin(movies['id'].tolist())]  
        l = list() 
        
        for item in  tempLinks["movieId"]:
                l.append (self.model_svd.predict(userId, item).est)
        
        tempData = pd.DataFrame({"title" :movies['title'],"est": l, "id" :movies['id']})
        tempData = tempData.sort_values('est', ascending=False)
        tempData = tempData[tempData["est"] >= 3]

        return tempData
    
    def recommender(self, title=None, userId=None, similarity_weight=0.9, top_n=1000):
        if self.model_name == 'Movie':
            data = self.model_data.reset_index()
            idx = data[data['title'] == title].index[0]
            cosine_sim = cosine_similarity(self.tfidf_matrix[int(idx)], self.tfidf_matrix)
            similarity = cosine_sim[0].T

            sim_data = pd.DataFrame(similarity, columns=['similarity'])
            final_data = pd.concat([data, sim_data], axis=1)

            final_data['final_score'] = final_data['score']*(1-similarity_weight) + final_data['similarity']*similarity_weight

            final_data_sorted = final_data.sort_values(by='final_score', ascending=False).head(top_n)
            self_index = final_data_sorted[final_data_sorted['title'] == title].index
            final_data_sorted.drop(self_index, inplace=True)
            remove_indices = final_data_sorted[final_data_sorted['similarity'] < 0.01].index
            final_data_sorted.drop(remove_indices, inplace=True)
            movies_indices = final_data_sorted.index.tolist()

            posters = []
            for i in movies_indices[:10]:
                posters.append(self.api_poster(self.model_data['id'].iloc[i]))

            names = []
            for i in movies_indices[:10]:
                names.append(self.model_data['title'].iloc[i])
            
            return names, posters, movies_indices
        elif self.model_name == 'User':
            tempRatings = self.model_ratings[self.model_ratings['userId'] == userId]
            tempRatings = tempRatings[tempRatings["rating"] >= 3.5]
            
            tempLinks = self.model_id[self.model_id['tmdbId'].isin(tempRatings['movieId'].tolist())]  
            titlesData = self.model_data[self.model_data["id"].isin(tempLinks["tmdbId"])]
            
            resultDataFrame = pd.DataFrame()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in titlesData["title"]:
                    futures.append(executor.submit(self.hybrid_svd, userId, i))
                for future in concurrent.futures.as_completed(futures):
                    try:
                        resultDataFrame = pd.concat([future.result(), resultDataFrame], ignore_index = True)
                    except:
                        print('connect time out')
            return resultDataFrame.sort_values('est', ascending=False)
        elif self.model_name == 'User & Movie':
            return self.hybrid_svd(userId=userId, title=title)
