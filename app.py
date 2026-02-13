import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import urllib.parse

@st.cache_data
def load_data():
    df = pd.read_csv('songs.csv')
    column_mapping = {
            'Title': 'track_name',
            'Artist': 'artists',
            'Top Genre': 'track_genre',
            'Beats Per Minute (BPM)': 'tempo',
            'Energy': 'energy',
            'Danceability': 'danceability',
            'Valence': 'valence',
            'Acousticness': 'acousticness',
            'Speechiness': 'speechiness',
            'Popularity': 'popularity'
    }
        
    df = df.rename(columns=column_mapping)
    df = df.drop_duplicates(subset=['track_name', 'artists'])
        
    return df.reset_index(drop=True)

try:
    df = load_data()
    df = df.fillna(0)
except FileNotFoundError:
    st.error("'songs.csv' not found")
    st.stop()


feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'speechiness']

missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    st.error(f"Your dataset is missing these columns: {missing_cols}")
    st.stop()

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

df_scaled['tempo'] = df_scaled['tempo'] * 1.2
df_scaled['speechiness'] = df_scaled['speechiness'] * 1.5

st.set_page_config(page_title="Vibe Matcher", page_icon="🎵", layout="wide")

st.markdown("""
<style>
.stButton>button { width: 100%; background-color: #1DB954; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🎵 Spotify Vibe Matcher")
st.write("Find songs with the same mathematical 'vibe'.")

# Search Bar
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("🔍 Search for a song you love:", placeholder="e.g. Yellow")

# Filter options based on search
options = []
if search_query:
    # Case-insensitive search
    mask = df['track_name'].str.contains(search_query, case=False, na=False)
    filtered_df = df[mask].sort_values(by='popularity', ascending=False).head(10) # Show top 10 matches
    filtered_df['display_name'] = filtered_df['track_name'] + " - " + filtered_df['artists']
    
    options = filtered_df['display_name'].values

# Song Selector
if len(options) > 0:
    selected_song = st.selectbox("Select the specific version:", options)
else:
    selected_song = None
    if search_query:
        st.warning("No popular songs found with that name. Try another!")

if st.button("Find Matches") and selected_song:
    selected_track, selected_artist = selected_song.rsplit(' - ', 1)
    
    # Find the specific row that matches BOTH name and artist
    song_row = df[(df['track_name'] == selected_track) & (df['artists'] == selected_artist)]
    
    if len(song_row) == 0:
        st.error("Could not find that specific song in the database. Please try again.")
        st.stop()
        
    song_index = song_row.index[0]
    song_vector = df_scaled.loc[song_index, feature_cols].values.reshape(1, -1)
    selected_genre = song_row['track_genre'].values[0] if 'track_genre' in df.columns else None
    # Calculate Similarity against the entire dataset
    similarity_scores = cosine_similarity(song_vector, df_scaled[feature_cols])
    
    final_scores = similarity_scores[0].copy()
    
    # Sort by NEW scores
    similar_indices = final_scores.argsort()[-11:][::-1]
    
    # Display Results
    st.subheader(f"If you like '{selected_track}', you might like:")
    cols = st.columns(5)
    
    count = 0
    for i in similar_indices:
        if i == song_index: continue # Skip the song itself
        
        match = df.iloc[i]
        score = final_scores[i]
        
        with cols[count % 5]:
            # Generate Spotify Link
            query = f"{match['track_name']} {match['artists']}"
            safe_query = urllib.parse.quote(query)
            spotify_url = f"https://open.spotify.com/search/{safe_query}"
            
            with st.container(border=True):
                st.write(f"**{match['track_name']}**")
                st.caption(f"{match['artists']}".replace(";", ", "))
                
                genre_display = match['track_genre'] if 'track_genre' in match else "Unknown"
                st.info(f"Genre: {genre_display}")
                
                st.progress(int(score*100), text=f"Match: {int(score*100)}%")
                
                # The "Listen" Link
                st.link_button("Listen on Spotify", spotify_url)
            
        count += 1
        if count >= 5: break