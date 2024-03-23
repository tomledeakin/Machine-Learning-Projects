import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def clean_genres(genres):
    return genres.replace('|', ' ').replace('-', '')

data = pd.read_csv("movie_data/movies.csv", sep="\t", encoding="latin-1", usecols=["title", "genres"])
data['genres'] = data['genres'].apply(clean_genres)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['genres'])
tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out(), index=data.title)
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, columns=data.title, index=data.title)

top_n = 20
input_movie = "Maximum Risk (1996)"

result = cosine_sim_df.loc[input_movie, :]
result = result.sort_values(ascending=False)
result = result[:top_n]