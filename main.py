import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocessing
movies = movies.loc[:,["movieId","title"]]
ratings = ratings.loc[:,["userId","movieId","rating"]]
data = pd.merge(ratings,movies,on='movieId')

# Compute number of ratings given to each movie
rating_count = (data.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])

# Combine total rating count with the original data
rating_with_totalRatingCount = data.merge(rating_count, left_on = 'title', right_on = 'title', how = 'left')

# Filter out less popular movies
popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

# Create matrix with movies as columns and user ids as rows
movie_features_df = rating_popular_movie.pivot_table(index='userId',columns='title',values='rating').fillna(0)

# Compute similarity between movies
movie_similarity = cosine_similarity(movie_features_df.T)

def recommend_movies(movie_input):
    idx = movies.loc[movies['title'].isin([movie_input])]
    idx = idx.index
    similarity_score = pd.Series(movie_similarity[idx[0]])
    similarity_score.index = movies.index
    top_10_indexes = similarity_score.nlargest(11).index
    recommended_movies = movies.loc[top_10_indexes]
    return recommended_movies

print(recommend_movies('Pocahontas'))
