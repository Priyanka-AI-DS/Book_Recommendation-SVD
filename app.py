import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Book Recommender", layout="wide")

st.title("📚 AI Book Recommendation System")

# Load saved files
model = joblib.load("svd_model.pkl")
books = pd.read_csv("books_clean.csv")
data = pd.read_csv("interactions_df.csv")

users = data["User-ID"].unique()
selected_user = st.selectbox("Select User", users)

if st.button("Recommend Top 10 Books"):

    seen_books = set(
        data[data["User-ID"] == selected_user]["ISBN"]
    )

    all_books = set(data["ISBN"].unique())
    unseen_books = list(all_books - seen_books)

    predictions = []

    for isbn in unseen_books:
        pred = model.predict(selected_user, isbn)
        predictions.append((isbn, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_10 = predictions[:10]

    result_df = pd.DataFrame(top_10, columns=["ISBN", "Predicted Rating"])
    result_df = result_df.merge(
        books[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN",
        how="left"
    )

    st.dataframe(
        result_df[["Book-Title", "Book-Author", "Predicted Rating"]],
        use_container_width=True
    )