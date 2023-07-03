import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import models

def userRecommernder(userID ,choice):

    model = models()

    newCredits = model.credits

    tempDataFrame = newCredits[[newCredits['userId'] == userID]]
    tempDataFrame = tempDataFrame[tempDataFrame['rating'] >= 4.0]
    moviedData = model.movies
    new_movie_ids = tempDataFrame['movieId'].tolist()
    filtered_new_movie = moviedData[moviedData['id'].isin(new_movie_ids)]

    result = list()

    for item in filtered_new_movie["original_title"]:
        if choice == 1:
            result.append(model.combineRecommenderModel(item ,2))
        elif choice == 2:
            result.append(model.contentBasedRecommenderModel(item ,2))
        else:
            result.append(model.hybridImdbsRecommenderModel(item ,2))

    return result






