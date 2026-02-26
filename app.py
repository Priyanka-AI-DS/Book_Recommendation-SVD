import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Book Recommendation System")

st.title("📚 Book Recommendation System")
st.write("AI-Based Collaborative Filtering using SVD")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_books():
    return pd.read_csv("books_clean.csv")

books = load_books()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with open("svd_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------------
# USER INPUT
# -------------------------------
user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Get Recommendations"):

    try:
        book_ids = books["Book-ID"].unique()
        predictions = []

        for book_id in book_ids:
            pred = model.predict(user_id, book_id)
            predictions.append((book_id, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_books = predictions[:5]

        st.subheader("Top 5 Recommended Books")

        for book_id, rating in top_books:
            book_title = books[books["Book-ID"] == book_id]["Book-Title"].values[0]
            st.write(f"📖 {book_title}  (Predicted Rating: {round(rating,2)})")

    except Exception as e:
        st.error("Something went wrong. Please check User ID.")
        st.write(e)