{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "70c51cc9-4093-4e16-9996-d38bc8be0b34",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "st.title(\"🎧 Spotify Song Recommender\")\n",
    "\n",
    "# Load data\n",
    "df = pd.read_csv(\"spotify_dataset.csv\")\n",
    "df = df.drop_duplicates(subset=['track_name', 'track_artist'])\n",
    "df.dropna(inplace=True)\n",
    "\n",
    "TRACK_COL = 'track_name'\n",
    "ARTIST_COL = 'track_artist'\n",
    "\n",
    "# Numeric features\n",
    "numeric_cols = df.select_dtypes(include=np.number).columns\n",
    "X = df[numeric_cols]\n",
    "\n",
    "# Scale\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "def recommend_songs(song_name, n=5):\n",
    "    idx = df[df[TRACK_COL] == song_name].index[0]\n",
    "    sim_scores = cosine_similarity(\n",
    "        X_scaled[idx].reshape(1, -1),\n",
    "        X_scaled\n",
    "    ).flatten()\n",
    "    similar_indices = sim_scores.argsort()[::-1][1:n+1]\n",
    "    return df.loc[similar_indices, [TRACK_COL, ARTIST_COL]]\n",
    "\n",
    "# UI\n",
    "song = st.selectbox(\"Select a song\", df[TRACK_COL].unique())\n",
    "\n",
    "if st.button(\"Recommend\"):\n",
    "    results = recommend_songs(song, 5)\n",
    "    st.dataframe(results)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7e99887-81ce-4130-9345-b2f890a89d82",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
