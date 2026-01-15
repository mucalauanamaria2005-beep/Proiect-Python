import os
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



#Cai catre fisierele csv


BASE_DIR = os.path.dirname(__file__)

CREDITS_PATH = os.path.join(BASE_DIR, "data", "credits.csv")
MOVIES_PATH = os.path.join(BASE_DIR, "data", "movies.csv")



#Incarcare date


credits_df = pd.read_csv(CREDITS_PATH)
movies_df = pd.read_csv(MOVIES_PATH)

movies = movies_df.merge(credits_df, on="title")
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


#Functii auxiliare de procesare


def convert_list(text):
    try:
        L = ast.literal_eval(text)
    except:
        return []
    return [i["name"] for i in L]


def convert_cast(text):
    try:
        L = ast.literal_eval(text)
    except:
        return []
    return [i["name"] for i in L[:3]]


def get_director(text):
    try:
        L = ast.literal_eval(text)
    except:
        return ""
    for i in L:
        if i["job"].lower() == "director":
            return i["name"]
    return ""


#Procesare coloane

movies["genres"] = movies["genres"].apply(convert_list)
movies["keywords"] = movies["keywords"].apply(convert_list)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["director"] = movies["crew"].apply(get_director)

movies.drop(columns=["crew"], inplace=True)
movies["overview"] = movies["overview"].fillna("")



#Tags pentru similaritate generala

movies["tags"] = (
    movies["overview"] + " "
    + movies["genres"].apply(lambda x: " ".join(x)) + " "
    + movies["keywords"].apply(lambda x: " ".join(x)) + " "
    + movies["cast"].apply(lambda x: " ".join(x)) + " "
    + movies["director"]
).str.lower()

#Functie pentru similaritate


def build_similarity(column, max_features=5000):
    text_data = column.apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(text_data).toarray()
    return cosine_similarity(vectors)


#Matrice de similaritate


# General
cv_general = CountVectorizer(max_features=5000, stop_words="english")
vectors_general = cv_general.fit_transform(movies["tags"]).toarray()
similarity_general = cosine_similarity(vectors_general)

# Pe criterii
similarity_gen = build_similarity(movies["genres"])
similarity_tema = build_similarity(movies["keywords"])
similarity_actori = build_similarity(movies["cast"])
similarity_regizor = build_similarity(movies["director"])


#Selectarea matricei dupa criteriu


def get_similarity_matrix(criterion):
    criterion = criterion.lower().strip()

    if criterion == "gen":
        return similarity_gen
    elif criterion == "tema":
        return similarity_tema
    elif criterion == "actori":
        return similarity_actori
    elif criterion == "regizor":
        return similarity_regizor
    else:
        return similarity_general   # default: general


#Functia finala folosita de django


def recommend(movie_title, criterion="general", top_n=5):
    movie_title = movie_title.lower()
    titles_lower = movies["title"].str.lower()

    if movie_title not in titles_lower.values:
        return []

    idx = titles_lower[titles_lower == movie_title].index[0]
    similarity_matrix = get_similarity_matrix(criterion)

    distances = list(enumerate(similarity_matrix[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in distances[1:top_n + 1]:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations
