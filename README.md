# 🎵 Spotify Vibe Matcher

A precision-focused music recommendation system that uses **Content-Based Filtering** and **Weighted Feature Vectors** to find songs with a mathematically identical "vibe."

## Overview
Standard recommendation algorithms often fail to distinguish between songs that *look* similar mathematically but *sound* completely different (e.g., a lyrical Rap song vs. an instrumental EDM track).

This project solves that by implementing a **Weighted Content-Based System**:
1.  **Curated Dataset:** Uses a "Gold Standard" dataset of the Top 2000 tracks of all time to eliminate noise, covers, and low-quality data.
2.  **Vector Search:** Calculates Cosine Similarity on high-dimensional audio feature vectors (Tempo, Energy, Valence).
3.  **Feature Weighting:** Applies custom coefficients to specific features like `Speechiness` and `Tempo` to enforce stricter matching on vocal style and rhythm.

## Tech Stack
* **Core:** Python 3.10+
* **Data Science:** Pandas, Scikit-Learn (Cosine Similarity, MinMaxScaler)
* **Frontend:** Streamlit
* **Data Source:** [Spotify Top 2000s Mega Dataset](https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset) (Curated Hits)

## Key Features
* **Weighted Feature Engineering:** Implements domain-specific weights:
    * **Tempo (1.2x):** Prioritizes matching the exact BPM/Speed of the track.
    * **Speechiness (1.5x):** heavily penalizes matches that differ in vocal density (separating Rap from Pop).
* **Fuzzy Search:** Intelligent string matching allows users to find specific tracks even if they type only part of the name.
* **Deep Linking:** Generates direct Spotify links for immediate listening.

## How to Run / View

**Option 1: Live Demo (Recommended)**

This project is hosted on Streamlit. You can access the live version here:

**➡️ [Click to view Live Demo](https://vibematcher.streamlit.app/)**

**Option 2: Run Locally**

If you want to explore the code on your own machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/coooow/VibeMatcher.git](https://github.com/coooow/VibeMatcher.git)
    cd vibematcher
    ```

2.  **Install Dependencies**
    ```bash
    pip install streamlit pandas scikit-learn
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## The Math Behind It
The core recommendation engine operates in two stages:

1.  **Normalization & Weighting:**
    Raw audio features are scaled to a `0-1` range using MinMax Scaling. We then apply scalar multiplication to prioritize specific dimensions.
    
    > $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} \times W_{feature}$$
    *(Where $W_{speechiness} = 1.5$ and $W_{tempo} = 1.2$)*

2.  **Cosine Similarity:**
    We calculate the cosine of the angle between the user's song vector ($A$) and every other song vector ($B$) in the database.
    
    > $$\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$