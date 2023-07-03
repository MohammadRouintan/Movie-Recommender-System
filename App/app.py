import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from PIL import Image
from io import BytesIO

movies_meta = pd.read_csv('data/MetadataBasedRecommenderData.csv')
movies_meta['combine'] = movies_meta['combine'].fillna('')
tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_meta['combine'])
movies_list = pd.read_csv('data/movies_list.csv')
movies_list = movies_list.reset_index()
titles = movies_list['title']
ids = movies_list['id']

def calculate_tfidf(data):
    tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data)
    return tfidf_matrix

def api_poster(moive_id):
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

# tab1, tab2, tab3 = st.tabs(['Model1', 'Model2', 'Model3'])

# with tab1:
st.header('Movie Recommender System')
selected_movie1 = st.selectbox(
    "Type or select first movie from the dropdown",
    movies_list['title'], key=1
)

selected_movie2 = st.selectbox(
    "Type or select second movie from the dropdown",
    movies_list['title'], key=2
)

selected_movie3 = st.selectbox(
    "Type or select third movie from the dropdown",
    movies_list['title'], key=3
)

if st.button('Recommend', key=10):
    recommended_movie_names1, recommended_movie_posters1, movies_indices1 = recommender(selected_movie1)
    recommended_movie_names2, recommended_movie_posters2, movies_indices2 = recommender(selected_movie2)
    recommended_movie_names3, recommended_movie_posters3, movies_indices3 = recommender(selected_movie3)

    set1, indices1 = combine_recommendation(selected_movie1, movies_indices2 + movies_indices3)
    set2, indices2 = combine_recommendation(selected_movie2, movies_indices1 + movies_indices3)
    set3, indices3 = combine_recommendation(selected_movie3, movies_indices1 + movies_indices2)

    intersection_names = list(set1 & set2 & set3)
    indices = list(indices1 & indices2 & indices3)
    intresection_posters = []
    if intersection_names:
        for idx in indices:
            if idx != None:
                intresection_posters.append(api_poster(ids.iloc[idx]))

        cols = st.columns(5)
        for i in range(len(intersection_names)):
            if i < 20:
                if i % 5 == 0:
                    cols = st.columns(5)
                with cols[i - (i // 5)*5]:
                    st.text(intersection_names[i])
                    st.image(intresection_posters[i])
                


    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names1[0])
        st.image(recommended_movie_posters1[0])
    with col2:
        st.text(recommended_movie_names1[1])
        st.image(recommended_movie_posters1[1])
    with col3:
        st.text(recommended_movie_names1[2])
        st.image(recommended_movie_posters1[2])
    with col4:
        st.text(recommended_movie_names1[3])
        st.image(recommended_movie_posters1[3])
    with col5:
        st.text(recommended_movie_names1[4])
        st.image(recommended_movie_posters1[4])

    col5, col6, col7, col8, col9 = st.columns(5)
    with col5:
        st.text(recommended_movie_names2[0])
        st.image(recommended_movie_posters2[0])
    with col6:
        st.text(recommended_movie_names2[1])
        st.image(recommended_movie_posters2[1])
    with col7:
        st.text(recommended_movie_names2[2])
        st.image(recommended_movie_posters2[2])
    with col8:
        st.text(recommended_movie_names2[3])
        st.image(recommended_movie_posters2[3])
    with col9:
        st.text(recommended_movie_names2[4])
        st.image(recommended_movie_posters2[4])

    col10, col11, col12, col13, col14 = st.columns(5)
    with col10:
        st.text(recommended_movie_names3[0])
        st.image(recommended_movie_posters3[0])
    with col11:
        st.text(recommended_movie_names3[1])
        st.image(recommended_movie_posters3[1])
    with col12:
        st.text(recommended_movie_names3[2])
        st.image(recommended_movie_posters3[2])
    with col13:
        st.text(recommended_movie_names3[3])
        st.image(recommended_movie_posters3[3])
    with col14:
        st.text(recommended_movie_names3[4])
        st.image(recommended_movie_posters3[4])

# with tab2:
#     st.header('Movie Recommender System')
#     selected_movie1 = st.selectbox(
#         "Type or select first movie from the dropdown",
#         movies_list['title'], key=4
#     )

#     selected_movie2 = st.selectbox(
#         "Type or select second movie from the dropdown",
#         movies_list['title'], key=5
#     )

#     selected_movie3 = st.selectbox(
#         "Type or select third movie from the dropdown",
#         movies_list['title'], key=6
#     )

#     if st.button('Recommend', key=11):
#         recommended_movie_names1, recommended_movie_posters1 = recommender(selected_movie1)
#         recommended_movie_names2, recommended_movie_posters2 = recommender(selected_movie2)
#         recommended_movie_names3, recommended_movie_posters3 = recommender(selected_movie3)

#         col1, col2, col3, col4, col5 = st.columns(5)
#         with col1:
#             st.text(recommended_movie_names1[0])
#             st.image(recommended_movie_posters1[0])
#         with col2:
#             st.text(recommended_movie_names1[1])
#             st.image(recommended_movie_posters1[1])
#         with col3:
#             st.text(recommended_movie_names1[2])
#             st.image(recommended_movie_posters1[2])
#         with col4:
#             st.text(recommended_movie_names1[3])
#             st.image(recommended_movie_posters1[3])
#         with col5:
#             st.text(recommended_movie_names1[4])
#             st.image(recommended_movie_posters1[4])

#         col5, col6, col7, col8, col9 = st.columns(5)
#         with col5:
#             st.text(recommended_movie_names2[0])
#             st.image(recommended_movie_posters2[0])
#         with col6:
#             st.text(recommended_movie_names2[1])
#             st.image(recommended_movie_posters2[1])
#         with col7:
#             st.text(recommended_movie_names2[2])
#             st.image(recommended_movie_posters2[2])
#         with col8:
#             st.text(recommended_movie_names2[3])
#             st.image(recommended_movie_posters2[3])
#         with col9:
#             st.text(recommended_movie_names[4])
#             st.image(recommended_movie_posters[4])

#         col10, col11, col12, col13, col14 = st.columns(5)
#         with col10:
#             st.text(recommended_movie_names3[0])
#             st.image(recommended_movie_posters3[0])
#         with col11:
#             st.text(recommended_movie_names3[1])
#             st.image(recommended_movie_posters3[1])
#         with col12:
#             st.text(recommended_movie_names3[2])
#             st.image(recommended_movie_posters3[2])
#         with col13:
#             st.text(recommended_movie_names3[3])
#             st.image(recommended_movie_posters3[3])
#         with col14:
#             st.text(recommended_movie_names3[4])
#             st.image(recommended_movie_posters3[4])

# with tab3:
#     st.header('Movie Recommender System')

#     selected_movie1 = st.selectbox(
#         "Type or select first movie from the dropdown",
#         movies_list['title'], key=7
#     )

#     selected_movie2 = st.selectbox(
#         "Type or select second movie from the dropdown",
#         movies_list['title'], key=8
#     )

#     selected_movie3 = st.selectbox(
#         "Type or select third movie from the dropdown",
#         movies_list['title'], key=9
#     )

#     if st.button('Recommend', key=12):
#         recommended_movie_names1, recommended_movie_posters1 = recommender(selected_movie1)
#         recommended_movie_names2, recommended_movie_posters2 = recommender(selected_movie2)
#         recommended_movie_names3, recommended_movie_posters3 = recommender(selected_movie3)

#         col1, col2, col3, col4, col5 = st.columns(5)
#         with col1:
#             st.text(recommended_movie_names1[0])
#             st.image(recommended_movie_posters1[0])
#         with col2:
#             st.text(recommended_movie_names1[1])
#             st.image(recommended_movie_posters1[1])
#         with col3:
#             st.text(recommended_movie_names1[2])
#             st.image(recommended_movie_posters1[2])
#         with col4:
#             st.text(recommended_movie_names1[3])
#             st.image(recommended_movie_posters1[3])
#         with col5:
#             st.text(recommended_movie_names1[4])
#             st.image(recommended_movie_posters1[4])

#         col5, col6, col7, col8, col9 = st.columns(5)
#         with col5:
#             st.text(recommended_movie_names2[0])
#             st.image(recommended_movie_posters2[0])
#         with col6:
#             st.text(recommended_movie_names2[1])
#             st.image(recommended_movie_posters2[1])
#         with col7:
#             st.text(recommended_movie_names2[2])
#             st.image(recommended_movie_posters2[2])
#         with col8:
#             st.text(recommended_movie_names2[3])
#             st.image(recommended_movie_posters2[3])
#         with col9:
#             st.text(recommended_movie_names2[4])
#             st.image(recommended_movie_posters2[4])

#         col10, col11, col12, col13, col14 = st.columns(5)
#         with col10:
#             st.text(recommended_movie_names3[0])
#             st.image(recommended_movie_posters3[0])
#         with col11:
#             st.text(recommended_movie_names3[1])
#             st.image(recommended_movie_posters3[1])
#         with col12:
#             st.text(recommended_movie_names3[2])
#             st.image(recommended_movie_posters3[2])
#         with col13:
#             st.text(recommended_movie_names3[3])
#             st.image(recommended_movie_posters3[3])
#         with col14:
#             st.text(recommended_movie_names3[4])
#             st.image(recommended_movie_posters3[4])

    