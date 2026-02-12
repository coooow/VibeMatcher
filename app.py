import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_data():
    df = pd.read_csv('songs.csv')
    
    df = df.drop_duplicates(subset=['track_name', 'artists'])
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("'songs.csv' not found")
    st.stop()


feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

st.set_page_config(page_title="Vibe Matcher", page_icon="🎵", layout="wide")
st.title("Spotify Vibe Matcher")
st.write("Select a song, and I'll find 5 others with the same mathematical 'vibe' from our database.")

song_search = st.text_input("Search for a song:", "")

if song_search:
    filtered_df = df[df['track_name'].str.contains(song_search, case=False, na=False)]
    options = filtered_df['track_name'].values
else:
    options = []

selected_song = st.selectbox("Select match:", options)

if st.button("Find Matches") and selected_song:
    # Get the vector of the selected song
    song_index = df[df['track_name'] == selected_song].index[0]
    song_vector = df_scaled.loc[song_index, feature_cols].values.reshape(1, -1)
    
    # Calculate Similarity against the entire dataset
    similarity_scores = cosine_similarity(song_vector, df_scaled[feature_cols])
    
    # Sort and get top 5
    similar_indices = similarity_scores[0].argsort()[-6:][::-1]
    
    # Display Results
    st.subheader(f"If you like '{selected_song}', you might like:")
    cols = st.columns(5)
    
    count = 0
    for i in similar_indices:
        if i == song_index: continue # Skip the song itself
        
        match = df.iloc[i]
        score = similarity_scores[0][i]
        
        with cols[count]:
            # The Kaggle dataset usually has track_genre, but no image URL.
            # We'll show the Genre instead of the image.
            st.metric(label="Match Score", value=f"{int(score*100)}%")
            st.write(f"**{match['track_name']}**")
            st.caption(f"{match['artists']}")
            st.info(match['track_genre'] if 'track_genre' in match else "Genre Unknown")
            
        count += 1
        if count >= 5: break