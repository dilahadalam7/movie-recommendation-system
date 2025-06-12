# movie-recommendation-system
#a simple recommendation system that suggests movies based on user ratings or genres.
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Sample dataset
user_ratings = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
    'movie_id': [101, 102, 103, 101, 103, 101, 102, 103],
    'rating': [5, 4, 3, 4, 5, 3, 4, 5]
})

movies = pd.DataFrame({
    'movie_id': [101, 102, 103],
    'genre': ['Action, Thriller', 'Comedy, Romance', 'Action, Adventure']
})

# Create a sparse matrix for collaborative filtering
sparse_matrix = csr_matrix((user_ratings['rating'], (user_ratings['user_id'] - 1, user_ratings['movie_id'] - 101)))

# Build the collaborative filtering model
cf_model = NearestNeighbors(n_neighbors=2, metric='cosine')
cf_model.fit(sparse_matrix)

# Create a TF-IDF vectorizer for content-based filtering
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['genre'])

# Calculate similarity between movies for content-based filtering
cbf_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Calculate similarity between users for collaborative filtering
cf_similarity_matrix = cosine_similarity(sparse_matrix)

# Get recommendations for a user
def get_recommendations(user_id):
    # Get similar users using collaborative filtering
    distances, indices = cf_model.kneighbors(sparse_matrix[user_id - 1], n_neighbors=2)
    similar_users = indices[0][1:]

    # Get movies rated by similar users
    recommended_movies = []
    for similar_user in similar_users:
        rated_movies = user_ratings[user_ratings['user_id'] == similar_user + 1]['movie_id'].values
        for movie in rated_movies:
            # Calculate similarity between movies using content-based filtering
            movie_index = movies[movies['movie_id'] == movie].index[0]
            similarity_scores = list(enumerate(cbf_similarity_matrix[movie_index]))
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            recommended_movies.append(movies.iloc[similarity_scores[1][0]]['movie_id'])

    # Remove duplicates and return recommended movies
    return list(set(recommended_movies))

# Get hybrid recommendations for a user
def get_hybrid_recommendations(user_id):
    # Get collaborative filtering recommendations
    cf_recommendations = get_recommendations(user_id)

    # Get content-based filtering recommendations
    cbf_recommendations = []
    user_rated_movies = user_ratings[user_ratings['user_id'] == user_id]['movie_id'].values
    for movie in user_rated_movies:
        movie_index = movies[movies['movie_id'] == movie].index[0]
        similarity_scores = list(enumerate(cbf_similarity_matrix[movie_index]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        cbf_recommendations.append(movies.iloc[similarity_scores[1][0]]['movie_id'])

    # Combine and return hybrid recommendations
    return list(set(cf_recommendations + cbf_recommendations))

print(get_hybrid_recommendations(1))
