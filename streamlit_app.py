import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load data
movies = pd.read_csv(r"D:\cc\ml-latest-small\movies.csv")
ratings = pd.read_csv(r"D:\cc\ml-latest-small\ratings.csv")

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset = final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

from fuzzywuzzy import process

def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_name = movie_name.lower()  # Convert input to lowercase
    print("Input movie name:", movie_name)
    
    # Perform fuzzy string matching to find similar movie names
    matching_movies = process.extract(movie_name, movies['title'].str.lower(), limit=5)
    print("Matching movies:", matching_movies)
    
    # Filter movies based on similarity score
    movie_list = movies[movies['title'].str.lower().isin([match[0] for match in matching_movies])]
    
    if not movie_list.empty:
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
        return df
    else:
        return "No similar movies found. Please check your input"



st.title("Movie Recommendation System")

movie_name = st.text_input("Enter a movie name:")
if st.button("Get Recommendations"):
    recommendations = get_movie_recommendation(movie_name)
    st.write(recommendations)
