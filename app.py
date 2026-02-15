import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Spotify Recommender", layout="centered")
st.title("🎧 Spotify Song Recommender")

@st.cache_data
def load_data():
    df = pd.read_csv("spotify_dataset.csv")
    df = df.drop_duplicates(subset=["track_name", "track_artist"])
    df.dropna(inplace=True)
    return df

@st.cache_resource
def scale_features(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    return scaler.fit_transform(df[numeric_cols])

df = load_data()
X_scaled = scale_features(df)

TRACK_COL = "track_name"
ARTIST_COL = "track_artist"

def recommend_songs(song_name, n=5):
    idx = df[df[TRACK_COL] == song_name].index[0]
    scores = cosine_similarity(
        X_scaled[idx].reshape(1, -1),
        X_scaled
    ).flatten()
    indices = scores.argsort()[::-1][1:n+1]
    return df.loc[indices, [TRACK_COL, ARTIST_COL]]

song = st.selectbox("Select a song", df[TRACK_COL].unique())

if st.button("Recommend"):
    st.dataframe(recommend_songs(song))
