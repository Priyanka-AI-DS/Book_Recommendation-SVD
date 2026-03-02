import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Data
# -----------------------------
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")

# Load Trained Model
model = joblib.load("models/svd_model.pkl")

st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("📚 Book Recommendation System")
st.write("Get personalized book recommendations using SVD Model")

# -----------------------------
# User Input
# -----------------------------
user_id = st.number_input("Enter User ID", min_value=1)

# -----------------------------
# Recommendation Logic
# -----------------------------
if st.button("Get Recommendations"):

    if user_id not in ratings['User-ID'].unique():
        st.error("User ID not found in dataset")
    else:
        user_books = ratings[ratings['User-ID'] == user_id]['ISBN'].tolist()
        all_books = books['ISBN'].tolist()

        unseen_books = list(set(all_books) - set(user_books))

        predictions = []

        for book in unseen_books:
            pred = model.predict(user_id, book)
            predictions.append((book, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top5 = predictions[:5]

        st.subheader("⭐ Top 5 Recommended Books")

        for book_id, rating in top5:
            book_info = books[books['ISBN'] == book_id].iloc[0]

            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(book_info['Image-URL-M'])

            with col2:
                st.write(f"### {book_info['Book-Title']}")
                st.write(f"Author: {book_info['Book-Author']}")
                st.write(f"Predicted Rating: {round(rating,2)}")
