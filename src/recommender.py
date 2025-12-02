import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from urllib.parse import quote

class MovieRecommender:
    def __init__(self, csv_path: str):
        # Load dataset
        self.movies = pd.read_csv(csv_path)

        # Required columns for TMDB dataset
        needed_columns = ['id', 'title', 'genres', 'overview']
        for col in needed_columns:
            if col not in self.movies.columns:
                raise ValueError(
                    f"Column '{col}' not found in CSV. Available columns: {self.movies.columns.tolist()}"
                )
        self.movies = self.movies[needed_columns].copy()

        # Parse TMDB 'genres' JSON-like strings into plain text labels
        def parse_genres(genres_str):
            if pd.isna(genres_str) or genres_str == "[]":
                return ""
            try:
                genres = ast.literal_eval(genres_str)
                if isinstance(genres, list):
                    names = [g.get("name", "") for g in genres if isinstance(g, dict)]
                    return " ".join(names)
                return ""
            except (ValueError, SyntaxError):
                return ""

        self.movies['genres_parsed'] = self.movies['genres'].apply(parse_genres)

        # Handle missing overviews
        self.movies['overview'] = self.movies['overview'].fillna('')

        # Combined content: parsed genres + overview
        self.movies['content'] = (
            self.movies['genres_parsed'].astype(str) + " " +
            self.movies['overview'].astype(str)
        )

        # Store title list for suggestions
        self.title_list = self.movies['title'].tolist()

        # TF-IDF with better config
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies['content'])

        # Cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Map movie titles to index (case-insensitive)
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies['title'].str.lower()
        ).drop_duplicates()

    def _get_index_from_title(self, title: str):
        title = title.strip().lower()
        return self.indices.get(title, None)

    def get_title_suggestions(self, query: str, top_n: int = 5):
        if not query:
            return []

        lower_to_original = {t.lower(): t for t in self.title_list}
        matches_lower = get_close_matches(
            query.lower(),
            list(lower_to_original.keys()),
            n=top_n,
            cutoff=0.4
        )
        return [lower_to_original[m] for m in matches_lower]

    def recommend(self, movie_title: str, num_recommendations: int = 5):
        idx = self._get_index_from_title(movie_title)

        if idx is None:
            suggestions = self.get_title_suggestions(movie_title, top_n=5)
            return {
                "found": False,
                "input_title": movie_title,
                "message": "Movie not found in database.",
                "suggestions": suggestions,
                "recommendations": []
            }

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]

        movie_indices = [i[0] for i in sim_scores]
        scores = [s[1] for s in sim_scores]

        results = []
        for i, score in zip(movie_indices, scores):
            row = self.movies.iloc[i]

            query = quote(str(row['title']))
            imdb_url = f"https://www.imdb.com/find/?q={query}&s=tt&ttype=ft"

            results.append({
                "title": row['title'],
                "genres": row['genres_parsed'],
                "overview": row['overview'],
                "similarity": float(score),
                "url": imdb_url
            })


        return {
            "found": True,
            "input_title": self.movies.iloc[idx]['title'],
            "message": "Success",
            "suggestions": [],
            "recommendations": results
        }
