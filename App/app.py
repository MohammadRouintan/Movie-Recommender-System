import streamlit as st
import requests
import numpy as np
import pandas as pd
from model import Model
import concurrent.futures


movies_list = pd.read_csv('data/movies_list.csv')
movies_list = movies_list.reset_index()

st.markdown('<div style="text-align: center;"><h1>Movie Recommender System</h1></div>', unsafe_allow_html=True)

selected_user = st.text_input("Type User Id Between 1-671", key=0)
selected_movie1 = st.selectbox(
    "Type or select first movie",
    movies_list['title'], key=1
)

selected_movie2 = st.selectbox(
    "Type or select second movie",
    movies_list['title'], key=2
)

selected_movie3 = st.selectbox(
    "Type or select third movie",
    movies_list['title'], key=3
)


c1, c2, c3 = st.columns(3)
number = 0
with c1:
    if st.button('Recommend By Movie', key=10):
        number = 1

with c2:
    if st.button('Recommend By User', key=11):
        number = 2

with c3:
    if st.button('Recommend By User & Movie', key=12):
        number = 3

def correct_userId(string):
    try:
        user_id = int(string)
        if 1 <= user_id <= 671:
            return True
        else:
            return False
    except:
        return False

if number == 1:
    model = Model('Movie')
    pool1 = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    future1 = pool1.submit(model.recommender, selected_movie1)
    future2 = pool1.submit(model.recommender, selected_movie2)
    future3 = pool1.submit(model.recommender, selected_movie3)
    recommended_movie_names1, recommended_movie_posters1, movies_indices1 = future1.result()
    recommended_movie_names2, recommended_movie_posters2, movies_indices2 = future2.result()
    recommended_movie_names3, recommended_movie_posters3, movies_indices3 = future3.result()
    pool1.shutdown(wait=True)

    pool2 = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    future1 = pool2.submit(model.combine_recommendation, selected_movie1, movies_indices2 + movies_indices3)
    future2 = pool2.submit(model.combine_recommendation, selected_movie2, movies_indices1 + movies_indices3)
    future3 = pool2.submit(model.combine_recommendation, selected_movie3, movies_indices1 + movies_indices2)
    set1, indices1 = future1.result()
    set2, indices2 = future2.result()
    set3, indices3 = future3.result()
    pool2.shutdown(wait=True)

    intersection_names = list(set1 & set2 & set3)
    indices = list(indices1 & indices2 & indices3)
    intresection_posters = []
    if intersection_names:
        st.markdown('<div style="text-align: center;"><h4>Combination Of Three Movie Recommended</h4></div>', unsafe_allow_html=True)
        for idx in indices:
            if idx != None:
                intresection_posters.append(model.api_poster(model.model_data['id'].iloc[idx]))

        cols = st.columns(5)
        for i in range(len(intersection_names)):
            if i < 20:
                if i % 5 == 0:
                    cols = st.columns(5)
                with cols[i - (i // 5)*5]:
                    st.text(intersection_names[i])
                    st.image(intresection_posters[i])
                
    st.markdown('<div style="text-align: center;"><h5>Recommended to First Movie</h5></div>', unsafe_allow_html=True)
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

    st.markdown('<div style="text-align: center;"><h5>Recommended to Second Movie</h5></div>', unsafe_allow_html=True)
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

    st.markdown('<div style="text-align: center;"><h5>Recommended to Third Movie</h5></div>', unsafe_allow_html=True)
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
elif number == 2:
    if not correct_userId(selected_user):
        st.markdown('<div style="text-align: center;"><h4>Please Enter Correct User Id</h4></div>', unsafe_allow_html=True)
    else:
        model = Model('User')
        movies = model.recommender(userId=int(selected_user))
        movies_names = movies['title'].values.tolist()
        movies_id = movies['id'].values.tolist()
        posters = []
        if movies_names:
            for idx in movies_id:
                if idx != None:
                    posters.append(model.api_poster(idx))

            cols = st.columns(5)
            for i in range(len(movies_names)):
                if i < 10:
                    if i % 5 == 0:
                        cols = st.columns(5)
                    with cols[i - (i // 5)*5]:
                        st.text(movies_names[i])
                        st.image(posters[i])
elif number == 3:
    if not correct_userId(selected_user):
        st.markdown('<div style="text-align: center;"><h4>Please Enter Correct User Id</h4></div>', unsafe_allow_html=True)
    else:
        model = Model('User & Movie')
        pool1 = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        future1 = pool1.submit(model.recommender, selected_movie1, int(selected_user))
        future2 = pool1.submit(model.recommender, selected_movie2, int(selected_user))
        future3 = pool1.submit(model.recommender, selected_movie3, int(selected_user))
        recommended_movie1 = future1.result()
        recommended_movie2 = future2.result()
        recommended_movie3 = future3.result()
        pool1.shutdown(wait=True)
        recommended_movie = pd.concat([recommended_movie1, recommended_movie2, recommended_movie3], ignore_index=True)
        movies_names = recommended_movie['title'].values.tolist()
        movies_id = recommended_movie['id'].values.tolist()
        posters = []
        if movies_names:
            for idx in movies_id:
                if idx != None:
                    posters.append(model.api_poster(idx))

            cols = st.columns(5)
            for i in range(len(movies_names)):
                if i < 10:
                    if i % 5 == 0:
                        cols = st.columns(5)
                    with cols[i - (i // 5)*5]:
                        st.text(movies_names[i])
                        st.image(posters[i])