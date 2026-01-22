
# Data Mining Project: Italian Rap Songs

This project was developed as part of the *Data Minin** course at the University of Pisa (UniPi) by [Emiliano Sescu](https://github.com/Faxatos), [Jaanis Fehling](https://github.com/jaanisfehling), and [Dieudonné Iyabivuze](https://github.com/Iyabivuz-e). It presents a comprehensive data mining analysis of Italian rap music, examining lyrical patterns, audio features, and regional characteristics through clustering, classification, and time series methodologies.

A high-level overview is provided below. Detailed methodological explanations and intermediate results are documented within the individual Jupyter notebooks, while the [project  report](https://github.com/jaanisfehling/italian-rap-songs/blob/main/report.pdf) presents more in-depth findings accompanied by visual analyses.

##  Methodology & Tech Stack

This phase transformed *11,166 tracks* published from *104 artists*, from raw scraped data into a high-quality dataset through auditing and logical refinement.

### 1. Data Understanding & Preparation

- **Audit & Cleaning:** We resolved data type mismatches (e.g., converting `year` and `popularity` from objects to numerical types), stripped invisible Unicode characters, and corrected multiple ID collisions.
- **Semantic Cleaning:** Over 1000 mislabeled Italian tracks (initially tagged as other languanges, mostly Polish) were corrected. Non-lyrical artifacts and instrumentals were filtered using regular expressions over lyrics.
- **Mixed-Strategy Imputation:** Missing artist geographic data was recovered via the *Nominatim* and *WikiData* APIs, while the *Spotify API* provided missing track release years.
- **NLP Verification:** To fill text-based gaps, we used a comparative validation process that selected *Regex* for token/sentence counts over *SpaCy* based on better alignment with the original data's logic.
- **Correlation & Redundancy:** We pruned redundant features like `rms` and `zcr` to minimize multicollinearity, retaining `loudness` and `rolloff` respectively due to their superior alignment with human auditory perception and lower noise sensitivity.

---

### 2. Feature Engineering

We engineered custom metrics to capture the stylistic signatures of Italian rap sub-genres and production techniques.

- **Lyrical Sophistication:** We developed metrics for delivery speed (*Words Per Minute*), rhythmic density (*Syllables Per Beat*), and content intensity (*Explicitness Density*).
- **Production & Audio:** New composite scores were created to distinguish high-energy tracks (*Audio Aggressiveness*), instrumental richness (*Harmonic Complexit**), and mix brightness (*Vocal Clarity*).
- **Contextual Metadata:** We calculated the *Collaboration Count* to measure structural variety and *Artist Relative Popularity* ($Z$-score) to identify breakout hits relative to an artist's typical performance.
- **Refinement:** Feature distributions were inspected to identify and correct artifacts (e.g., imposing a minimum threshold on _Vocal Clarity_ to prevent extreme outliers arising from tracks with near-zero loudness). Correlation analysis was then applied to ensure that newly introduced features were not excessively correlated with the original set.

---

### 3. Clustering Analysis

This phase evaluated whether the Italian rap dataset contained inherent substructures using various unsupervised learning algorithms. Continuous features were standardized to unit variance, and samples with missing values were removed prior to analysis.

-   **K-Means Clustering:** We utilized *K-Means++* for improved initialization and identified **$k=9$** as a local optimum based on the *Silhouette Score* and *Davies-Bouldin Index*.
-   **Data Distribution Findings:** Visualizations using *Principal Component Analysis (PCA)* and low evaluation scores (Silhouette Score $\approx 0.09$; Davies-Bouldin Index $\approx 1.6$) revealed that the data largely presents a single, mostly homogeneous "uniform blob" rather than well-separated clusters.
-   **Density-based Clustering (DBSCAN):** Optimal results were found at *$\epsilon=4.5$*. This method isolated one large cluster and a small, distinct group (Cluster 1) consisting primarily of older tracks from around 2008.
-   **Hierarchical Clustering:** Using *ward linkage* to achieve balanced partitions, we found an optimal number of clusters at *$k=4$*. However, high Davies-Bouldin indices confirmed that Agglomerative clustering struggled to find a balanced partition within the data's dense distribution. 
-   **Alternative Methods:**  *X-Means* and**K-Medoids* were tested but produced metrics similar to standard K-Means, further confirming the lack of clear inherent substructures in the dataset.

---

### 4. Predictive Modeling & XAI

This phase focused on classifying tracks into regional "Rap Schools" using a combination of lyrical and audio attributes. It began with **data preparation and feature selection**, comprising the following steps:

 -   **Macro-Zone Mapping:** Artists were categorized into four linguistically motivated schools: *Milan Drill*, *Roman Slang*, *South Dialect*, and *Lyrical Standard* .
-   **Lyrical Representation:** Lyrical content was processed using *TF-IDF* (limited to the top 2,000 terms) to capture regional slang and dialect markers .    
-   **Feature Selection:** We removed non-informative attributes and identifiers to prevent model leakage, focusing solely on audio characteristics and lyric tokens.

**Model Training & Evaluation:**
We utilized *3-fold Stratified Cross-Validation* to optimize the *macro F1-score*, ensuring performance was balanced across the imbalanced classes.
- **Algorithms:** Four models were evaluated: *LightGBM*, *XGBoost*, *Random Forest*, and *Linear SVM*
- **Best Model Performance:**  *LightGBM_refit* was the top-performing model, achieving a *Test F1-macro of 0.6040* and a *Test Accuracy of 0.6089*.

**Explainable AI (XAI) with SHAP:**
To ensure transparency, *SHAP (SHapley Additive exPlanations)* was used to interpret the LightGBM model.
- **Global Importance:** The model combined acoustic descriptors (e.g., track duration and loudness) with specific lyric tokens to distinguish schools.
- **Model Logic:** Lyrical tokens indicative of regional slang (e.g., "milano") were among the strongest predictors, confirming that the model successfully learned geographical lexical cues.

---

### 5. Time Series Analysis
This final phase involved the processing and analysis of raw audio signals to identify temporal patterns and artist-specific signatures. It began with **preprocessing and feature extraction**, comprising the following steps:

-   **Signal Cleaning:** Raw audio was loaded via *librosa*, with leading/trailing sequences quieter than 25 decibels trimmed and amplitudes peak-normalized .
-   **Temporal Framing:** Beat tracking was applied to calculate tempo and segment the audio into frames.
-   **Feature Set:** We extracted a multidimensional feature vector per frame consisting of *13 MFCCs*, *12 Chroma features*, and *1 onset feature*.

**Clustering & Motif Extraction:**
-   **Aggregated Clustering:** To simplify the time dimension, features were aggregated into a 62-dimensional vector per song5. K-Means++ identified **$k=2$** as the optimal (though poorly separated) partition.
-   **Motif Discovery:** We reduced Chroma features to a 1D entropy measure to extract 32-beat motifs 7.
-   **Repetition Analysis:** Cluster 1 exhibited a lower average motif distance (1.96) compared to Cluster 0 (2.38), indicating *significantly stronger repetition* in its tracks 8.

**Anomaly Detection & Shapelets:**
-   **Anomaly Identification:** Using Euclidean distance from cluster averages, we mapped "anomaly intensity" across 128 resampled datapoints . Results showed a consistent trend of fewer anomalies at the beginning and end of songs.
-   **Shapelet Classification:** A classifier was trained to distinguish between specific artists (e.g., Fedez vs. Fabri Fibra) using learned temporal patterns (shapelets).
-   **Performance:** The shapelet model achieved a *42% accuracy* and identified unique MFCC signatures that were highly dominant for specific artists (80% dominance in top matching songs).

---

### Project Structure
```
📂 .
├── 📂 clustering/
│   ├── 📒 density_based.ipynb
│   ├── 📒 hierarchical.ipynb
│   ├── 📒 kmeans.ipynb
│   └── 📒 xmeans_kmedoids.ipynb
├── 📂 data_understanding/
│   ├── 📒 artists_analysis.ipynb
│   ├── 🐍 artists_cleaning.py
│   ├── 📒 data_distribution.ipynb
│   ├── 📒 feature_eng_analysis.ipynb
│   ├── 🐍 feature_engineering.py
│   ├── 📒 tracks_analysis.ipynb
│   └── 🐍 tracks_cleaning.py
├── 📂 dataset/
│   ├── 📊 artists.csv
│   ├── 📊 cleaned_artists.csv
│   ├── 📊 cleaned_tracks.csv
│   ├── 📊 engineered_tracks.csv
│   └── 📊 tracks.csv
├── 📂 predictive_analysis/
│   └── 📒 predictive_analysis.ipynb
├── 📂 timeseries/
│   └── 📒 timeseries.ipynb
└── ⚙️ .gitignore
```
**Dataset Files:**

-   `artists.csv` and `tracks.csv` are the original datasets.
-   `cleaned_artists.csv` and `cleaned_tracks.csv` have data cleaning applied.
-   `engineered_tracks.csv` is the cleaned tracks dataset augmented with the new features described above.

##  Tech Stack

- **Data Processing:** pandas, numpy, scipy
- **NLP:** spacy, langdetect, TfidfVectorizer
- **Audio:** librosa
- **ML:** scikit-learn, xgboost, lightgbm, pyclustering
- **XAI:** shap
- **APIs:** Spotify, Nominatim, WikiData
