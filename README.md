# Lakshay Kumar 
(B.Tech , Computer Science and Engineering)

# Movie Recommendation System

A **content-based Movie Recommendation System** built using **Python, Flask, and Machine Learning**.

The system suggests films to a specified movie by analyzing **genres and plot resemblance** through **TF-IDF and Cosine Similarity**.



# Key Features

- Content-based movie recommendations (no user login required)

- Uses **TF-IDF (unigrams + bigrams)** for text vectorization

- **Cosine Similarity** for measuring similarity between movies

- Handles large real-world dataset (TMDB 5000 movies)

- Fuzzy search suggestions for incorrect movie names

- Modern and responsive UI

- Animated recommendation cards

- Movie cards that can be clicked lead, to IMDb search results pages


# How the Recommendation System Works

1. Movie **genres** and **overview text** are combined into a single content feature

2. Numerical vectors are created from text data through the use of **TF-IDF**

3. **Cosine Similarity** is computed among all movie vectors.

4. When a user inputs a movie title:

- The system identifies the closely corresponding title

- Suggests the **leading N movies with the similarity**

This method guarantees that suggestions rely on **movie content** proving efficient even in the absence of user interaction information.


# Dataset

- **TMDB 5000 Movie Dataset**

- Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

- Dataset utilized: `tmdb_5000_movies.csv`

> Note:

> The dataset is **not included** in this repository due to size limits.

> Please download it manually and place it inside the `data/` folder.

# Tech Stack

- **Python**

- **Flask**

- **pandas**

- **scikit-learn**

- **HTML & CSS** (custom UI)

- **Git & GitHub**

```text
## 📂 Folder Structure

```text
movie-recommendation-system/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ src/
│  └─ recommender.py
├─ templates/
│  ├─ base.html
│  └─ index.html
├─ static/
│  └─ css/
│     └─ style.css
└─ data/
   └─ tmdb_5000_movies.csv   (download from Kaggle)
```
# If you found this project useful, consider giving it a star!
