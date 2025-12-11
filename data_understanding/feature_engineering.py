import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def feature_engineering(input_file, output_file):
    print(f"Loading data from {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} was not found.")
        return

    df = pd.read_csv(input_file, sep=',')

    # 1. Lyrical Features
    
    # words_per_minute
    # n_tokens / duration in minutes
    df['words_per_minute'] = df['n_tokens'] / (df['duration_ms'] / 60000)

    # syllables_per_beat
    # n_tokens / (duration_minutes * bpm)
    df['syllables_per_beat'] = df['n_tokens'] / ((df['duration_ms'] / 60000) * df['bpm'])

    # explicitness_density
    # Ratio of swear words to total tokens
    df['explicitness_density'] = (df['swear_IT'].fillna(0) + df['swear_EN'].fillna(0)) / df['n_tokens']
    df['explicitness_density'] = df['explicitness_density'].fillna(0) #handle potential div by zero

    # 2. Audio Features

    # audio_aggressiveness
    # composite score of normalized intensity features
    audio_feats_agg = ['rolloff', 'loudness', 'flux', 'bpm']
    
    if all(col in df.columns for col in audio_feats_agg):
        agg_data = df[audio_feats_agg].copy()
        norm_agg = (agg_data - agg_data.min()) / (agg_data.max() - agg_data.min())
        df['audio_aggressiveness'] = norm_agg.mean(axis=1, skipna=False)
    else:
        print(f"Warning: Missing columns for audio_aggressiveness. Expected: {audio_feats_agg}")
        df['audio_aggressiveness'] = 0

    # harmonic_complexity
    # spectral_complexity * (1 - flatness)
    df['harmonic_complexity'] = df['spectral_complexity'] * (1 - df['flatness'])

    # vocal_clarity
    # rolloff / absolute loudness
    df['vocal_clarity'] = df['rolloff'] / df['loudness'].abs()

    # 3. Metadata Features

    # collab_count
    # number of artists in the 'featured_artists' column
    def count_featured(val):
        if pd.isna(val):
            return 0
        return len(str(val).split(','))

    if 'featured_artists' in df.columns:
        df['collab_count'] = df['featured_artists'].apply(count_featured)
    else:
        df['collab_count'] = 0

    # artist_rel_popularity
    # Z-score of popularity normalized by artist
    if 'name_artist' in df.columns and 'popularity' in df.columns:
        # create temp columns for vectorization
        artist_mean = df.groupby('name_artist')['popularity'].transform('mean')
        artist_std = df.groupby('name_artist')['popularity'].transform('std')
        
        df['artist_rel_popularity'] = (df['popularity'] - artist_mean) / artist_std
        df['artist_rel_popularity'] = df['artist_rel_popularity'].fillna(0) # Handle single-song artists (std=NaN)
    else:
        df['artist_rel_popularity'] = 0

    # 4. Cleanup
    
    # Columns to remove based on correlation analysis
    cols_to_remove = ['zcr', 'rms']
    df.drop(columns=[c for c in cols_to_remove if c in df.columns], inplace=True, errors='ignore')

    # ensure the requested new features are present
    requested_features = [
        'words_per_minute', 'syllables_per_beat', 'explicitness_density', 
        'audio_aggressiveness', 'harmonic_complexity',
        'vocal_clarity', 'collab_count', 'artist_rel_popularity'
    ]
    
    # verify creation
    missing_features = [f for f in requested_features if f not in df.columns]
    if missing_features:
        print(f"Warning: The following features could not be created: {missing_features}")

    # Save result
    print(f"Saving engineered dataset to {output_file}...")
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Define paths assuming script is run from the same level as the dataset or root
    # Adjust paths if your folder structure differs
    INPUT_FILENAME = '../dataset/cleaned_tracks.csv'
    OUTPUT_FILENAME = '../dataset/engineered_tracks.csv'
    
    feature_engineering(INPUT_FILENAME, OUTPUT_FILENAME)