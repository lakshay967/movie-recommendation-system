from flask import Flask, render_template, request
from src.recommender import MovieRecommender
import os

app = Flask(__name__)

# Path to dataset
CSV_PATH = os.path.join("data", "tmdb_5000_movies.csv")


# Initialize global recommender model once at startup
recommender = MovieRecommender(CSV_PATH)


# @app.route("/", methods=["GET", "POST"])

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error_message = None
    input_title = ""
    suggestions = []

    if request.method == "POST":
        input_title = request.form.get("movie_title", "").strip()

        if input_title:
            result = recommender.recommend(input_title, num_recommendations=5)

            if not result["found"]:
                error_message = f"Movie '{input_title}' not found in the database."
                suggestions = result.get("suggestions", [])
            else:
                input_title = result["input_title"]
                recommendations = result["recommendations"]
        else:
            error_message = "Please enter a movie title."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error_message=error_message,
        input_title=input_title,
        suggestions=suggestions,
    )


if __name__ == "__main__":
    app.run(debug=True)
