import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of movies
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'Inception', 'The Dark Knight', 'Interstellar', 'The Prestige'],
    'genre': ['Action Sci-Fi', 'Action Sci-Fi Thriller', 'Action Crime Drama', 'Adventure Drama Sci-Fi', 'Drama Mystery Sci-Fi'],
    'description': [
        'A computer hacker learns about the true nature of his reality and his role in the war against its controllers.',
        'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
        'When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
        'After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Combine genre and description for feature extraction
df['features'] = df['genre'] + " " + df['description']

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Convert the features into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Calculate cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on a movie title
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index
    if len(idx) == 0:
        return f"Movie titled '{title}' not found."
    idx = idx[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 5 most similar movies (excluding itself)
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return df['title'].iloc[movie_indices]

# Get user input
user_input = input("Enter a movie title: ")

# Recommend movies
recommended_movies = recommend_movies(user_input)
print("Movies similar to '{}':".format(user_input))
if isinstance(recommended_movies, str):
    print(recommended_movies)
else:
    print(recommended_movies.to_list())
