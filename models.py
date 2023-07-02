import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MinMaxScaler



class Model:

    def __int__(self):
        self.movies = pd.read_csv('data/MoviesMetadata.csv')
        self.links = pd.read_csv('data/links.csv')
        self.ratings = pd.read_csv('data/ratings.csv')
        self.credits = pd.read_csv('data/NewCredits.csv')
        self.keywords = pd.read_csv('data/NewKeywords.csv')


    def find_idx(title ,indices):
        idx = pd.Series(indices[title])
        return idx[0]

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False


    def find_unique(self,keywords_str):
        self.keywords_set = set()
        if not self.is_float(keywords_str):
            string_list = keywords_str.split(', ')
            for string in string_list:
                self.keywords_set.add(string)


    def find_director(self, job_crew, index ,movies_meta):
        if not self.is_float(job_crew):
            jobs = job_crew.split(', ')
            for job in jobs:
                if job == 'Director':
                    idx = jobs.index(job)
                    names = movies_meta.loc[index, 'name_crew']
                    if not self.is_float(names):
                        names = names.split(', ')
                        return names[idx]
                    else:
                        return np.nan
        return np.nan

    def count(self,keywords_str):
        self.keywords_dict = dict()
        if not self.is_float(keywords_str):
            string_list = keywords_str.split(', ')
            for string in string_list:
                self.keywords_dict[string] += 1

    def keywords_filtering(self,keywords):
        final_keywords = list()
        if not self.is_float(keywords):
            keywords_list = keywords.split(', ')
            for key in keywords_list:
                if key in self.keyword_unique:
                    final_keywords.append(key)
        return final_keywords

    def str_to_list(self,col):
        if not self.is_float(col):
            return col.split(', ')
        else:
            return []

    def collabrativeFilteringRecommenderModel(self):
        self.movies.drop(columns=['iso_3166_1_production_countries', 'id_production_companies', 'id_genres'], inplace=True)
        reader = Reader()
        ratings_data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], reader)
        svd = SVD()
        cross_validate(svd, ratings_data, measures=['RMSE'], cv=10)
        train = ratings_data.build_full_trainset()
        svd.fit(train)
        user_rating = pd.merge(self.ratings, self.movies, left_on='movieId', right_on='id', how='inner')
        user_rating = user_rating[['userId', 'movieId', 'rating', 'original_title']]
        user_ratings = user_rating.sort_values(by='userId')


    def contentBasedRecommenderModel(self, title ,size):
        movies = self.movies.drop(columns=['iso_3166_1_production_countries', 'id_production_companies', 'id_genres'])

        links = self.links[self.links['tmdbId'].notnull()]['tmdbId'].astype('int')

        movies_meta = movies[movies['id'].isin(links)]

        movies_meta['description'] = movies_meta['overview'] + movies_meta['tagline']
        movies_meta['description'] = movies_meta['description'].fillna('')

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(movies_meta['description'])
        cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

        movies_meta = movies_meta.reset_index()
        titles = movies_meta['title']

        indices = pd.Series(movies_meta.index, index=movies_meta['title'])

        idx = self.find_idx(title=title ,indices = indices)

        similarity_scores = list(enumerate(cosine_similarity[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:21]
        movie_indices = [i[0] for i in similarity_scores]

        return titles.iloc[movie_indices].head(size)

    def combineRecommenderModel(self ,title ,size):

        movies = self.movies.drop(columns=['iso_3166_1_production_countries', 'id_production_companies', 'id_genres'])

        keywords = self.keywords
        links = self.links
        credits = self.credits

        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = self.credits['id'].astype('int')

        movies = movies.merge(credits, on='id')
        movies = movies.merge(keywords, on='id')

        links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

        movies_meta = movies[movies['id'].isin(links)]
        movies_meta['director'] = None

        for i in range(movies_meta.shape[0]):
            movies_meta.loc[i, 'director'] = self.find_director(movies_meta.loc[i, 'job_crew'], i ,movies_meta)

        movies_meta['director'] = movies_meta['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
        movies_meta['director'] = movies_meta['director'].apply(lambda x: [x])

        movies_meta['name_keywords'].apply(self.find_unique)

        for keyword in self.keywords_set:
            self.keywords_dict[keyword] = 0

        movies_meta['name_keywords'].apply(self.count)

        self.keyword_unique = pd.Series(self.keywords_dict)

        self.keyword_unique = self.keyword_unique[self.keyword_unique > 1]

        stemmer = SnowballStemmer('english')

        movies_meta['name_keywords'] = movies_meta['name_keywords'].apply(self.keywords_filtering)
        movies_meta['name_keywords'] = movies_meta['name_keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
        movies_meta['name_keywords'] = movies_meta['name_keywords'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x])

        for i in range(movies_meta.shape[0]):
            if movies_meta.loc[i, 'name_genres'] == '[]':
                movies_meta.loc[i, 'name_genres'] = np.nan

        movies_meta['name_cast'] = movies_meta['name_cast'].apply(self.str_to_list)
        movies_meta['name_genres'] = movies_meta['name_genres'].apply(self.str_to_list)

        movies_meta['name_genres'] = movies_meta['name_genres'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x])
        movies_meta['name_cast'] = movies_meta['name_cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

        movies_meta['combine'] = movies_meta['name_keywords'] + movies_meta['name_cast'] + movies_meta['director'] + \
                                 movies_meta['name_genres']
        movies_meta['combine'] = movies_meta['combine'].apply(lambda x: ' '.join(x))

        count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        count_matrix = count.fit_transform(movies_meta['combine'])
        cosine_similarity = linear_kernel(count_matrix, count_matrix)

        movies_meta = movies_meta.reset_index()
        titles = movies_meta['title']
        indices = pd.Series(movies_meta.index, index=movies_meta['title'])

        idx = self.find_idx(title=title)
        similarity_scores = list(enumerate(cosine_similarity[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:21]
        movie_indices = [i[0] for i in similarity_scores]

        return titles.iloc[movie_indices].head(size)


    def hybridImdbsRecommenderModel(self ,title, similarity_weight=0.7, top_n=10):
        movies = self.movies
        keywords = self.keywords
        credits = self.credits

        movies['id'] = movies['id'].astype('object')
        df = movies.merge(keywords, on='id')
        movies['id'] = movies['id'].astype('int64')
        df = movies.merge(credits, on='id')

        df['original_language'] = df['original_language'].fillna('')
        df['runtime'] = df['runtime'].fillna(0)
        df['tagline'] = df['tagline'].fillna('')

        df.dropna(inplace=True)

        R = df['vote_average']
        v = df['vote_count']
        m = df['vote_count'].quantile(0.8)
        C = df['vote_average'].mean()

        df['weighted_average'] = (R * v + C * m) / (v + m)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['popularity', 'weighted_average']])
        weighted_df = pd.DataFrame(scaled, columns=['popularity', 'weighted_average'])

        weighted_df.index = df['original_title']
        weighted_df['score'] = weighted_df['weighted_average'] * 0.4 + weighted_df['popularity'].astype('float64') * 0.6
        weighted_df_sorted = weighted_df.sort_values(by='score', ascending=False)
        temp = ['original_language', 'original_title', 'overview', 'tagline', 'name_genres',
                'name_production_countries',
                'name_production_companies', 'name_crew', 'department_crew', 'job_crew', 'character_cast']
        hybrid_df = df[temp]

        hybrid_df['bag_of_words'] = ''
        hybrid_df['bag_of_words'] = hybrid_df[hybrid_df.columns[1:]].apply(lambda x: ' '.join(x), axis=1)
        hybrid_df.set_index('original_title', inplace=True)

        hybrid_df = hybrid_df[['bag_of_words']]

        hybrid_df = weighted_df_sorted[:10000].merge(hybrid_df, left_index=True, right_index=True, how='left')

        hybrid_df['bag_of_words'] = hybrid_df['bag_of_words'].fillna('')

        tfidf = TfidfVectorizer(analyzer="word", stop_words='english', ngram_range=(1, 2), min_df=0)
        tfidf_matrix = tfidf.fit_transform(hybrid_df['bag_of_words'])

        cos_sim = linear_kernel(tfidf_matrix)

        data = hybrid_df.reset_index()
        index_movie = data[data['original_title'] == title].index
        similarity = cos_sim[index_movie].T

        sim_df = pd.DataFrame(similarity, columns=['similarity'])
        final_df = pd.concat([data, sim_df], axis=1)

        final_df['final_score'] = final_df['score'] * (1 - similarity_weight) + final_df[
            'similarity'] * similarity_weight

        final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
        final_df_sorted.set_index('original_title', inplace=True)


        return final_df_sorted[['score', 'similarity', 'final_score']]






