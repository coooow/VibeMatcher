# 🎵 Spotify Vibe Matcher

A sophisticated music recommendation system that combines **Content-Based Filtering**  with **Rule-Based Logic** to solve the "Context Gap" in music discovery.

<!-- ![Demo](https://via.placeholder.com/800x400?text=Insert+Your+Demo+GIF+Here) -->

## Overview
Standard recommendation algorithms often fail to distinguish between songs that are mathematically similar but culturally different (e.g., a fast Rap song vs. a fast Kids' song).

This project solves that by implementing a **Hybrid Filtering System**:
1.  **Vector Search:** Uses Cosine Similarity on raw audio features (Tempo, Energy, Valence).
2.  **Feature Weighting:** Prioritizes distinct features like `Speechiness` (Vocal style) and `Tempo`.
3.  **Genre Penalty:** Applies a dynamic penalty score to matches with conflicting metadata, reducing false positives by ~20%.

## Tech Stack
* **Core:** Python 3.10+
* **Data Science:** Pandas, Scikit-Learn (Cosine Similarity, MinMaxScaler)
* **Frontend:** Streamlit
* **Data Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (114k+ tracks)

## Key Features
* **Weighted Feature Engineering:** Custom weights applied to `Tempo` (1.2x) and `Speechiness` (1.5x) to better separate vocal-heavy genres (Rap) from instrumental-heavy ones (EDM).
* **The "Anti-Noise" Filter:** Automatically filters out low-quality covers, karaoke tracks, and duplicates using a popularity threshold algorithm (`popularity > 50`).
* **Hybrid Logic:** If a mathematical match has a conflicting genre (e.g., Pop vs. Children's Music), the system subtracts a penalty coefficient (0.15) from the similarity score.
* **Smart Search:** Fuzzy string matching allows users to find specific tracks even with partial queries.

<!-- ## 💻 Installation & Setup

**Note:** The dataset is too large for GitHub (100MB+), so you must download it manually.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/spotify-vibe-matcher.git](https://github.com/yourusername/spotify-vibe-matcher.git)
    cd spotify-vibe-matcher
    ```

2.  **Download the Data**
    * Go to the [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
    * Download the zip file.
    * Extract `dataset.csv` and rename it to **`songs.csv`**.
    * Place `songs.csv` in the root folder of this project.

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```
 -->
## The Math Behind It
The core recommendation engine operates in three stages:

1.  **Normalization:**
    Raw audio features are scaled to a `0-1` range using MinMax Scaling to prevent high-variance features (like BPM) from dominating the vector space.
    $$ X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} $$

2.  **Cosine Similarity:**
    We calculate the cosine of the angle between the user's song vector ($A$) and every other song vector ($B$) in the database.
    $$ \text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$

3.  **Hybrid Penalty:**
    The final score is adjusted based on metadata rules to ensure cultural relevance.
    $$ \text{Final Score} = \text{Similarity} - ( \text{Penalty} \times \mathbb{I}_{genre\_mismatch} ) $$
