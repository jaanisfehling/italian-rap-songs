import pandas as pd
import numpy as np
import re
import spacy
import warnings
import ast
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from os import path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --- functions from track analysis' notebook ---

def load_spacy_models():
    models_tokenizers = {}
    models_pos_taggers = {}
    
    #lightweight spacy models
    try:
        nlp_it_light = spacy.load('it_core_news_sm', 
                                  disable=['parser', 'tagger', 'ner', 'lemmatizer'])
        nlp_it_light.add_pipe('sentencizer')
        models_tokenizers['it'] = nlp_it_light
        
        nlp_en_light = spacy.load('en_core_web_sm', 
                                  disable=['parser', 'tagger', 'ner', 'lemmatizer'])
        nlp_en_light.add_pipe('sentencizer')
        models_tokenizers['en'] = nlp_en_light
        
    except (IOError, ImportError):
        print("Error: spaCy lightweight models not found")
        return None, None

    #POS tagging models (slower since we keep tagger and parser enabled)
    try:
        models_pos_taggers['it'] = spacy.load('it_core_news_sm', 
                                             disable=['ner', 'lemmatizer'])
        
        models_pos_taggers['en'] = spacy.load('en_core_web_sm', 
                                             disable=['ner', 'lemmatizer'])
        
    except (IOError, ImportError):
        print("Error: spaCy POS models not found")
        return None, None

    return models_tokenizers, models_pos_taggers

def sort_splitted_by_comma(string):
    if pd.isna(string):
        return string
    ret = [a.strip() for a in string.split(',')]
    return ', '.join(sorted(ret))

def clean_contributor_lyrics(row, rescue_list):
    lyrics_pattern = re.compile(r'^\d+\s+Contributor(s)?.*?Lyrics(.*)', re.DOTALL)
    if pd.isna(row['lyrics']):
        return np.nan
        
    if row['title'] in rescue_list:
        match = lyrics_pattern.search(row['lyrics'])
        if match:
            rescued_text = match.group(2).strip()
            
            if not rescued_text: #if text after regex is empty
                return np.nan
            else:
                return rescued_text
        else:
            print(f"regex failed for target track: {row['title']}")
            return row['lyrics']
    else:
        #not a target track
        return row['lyrics']

def nullify_bad_lyrics(lyrics): #sets lyrics for instrumentals, metadata, and descriptions to NaN.
    if pd.isna(lyrics):
        return np.nan
    
    #nullify remaining "contributor"
    if 'Contributor' in lyrics:
        return np.nan
    
    return lyrics

prod_pattern = re.compile(r'La produzione.*', re.IGNORECASE | re.DOTALL)

def remove_production_only_lyrics(row):
    if pd.isna(row['lyrics']):
        return np.nan

    # Use row['n_sentences'] directly (assuming it exists in input CSV)
    # Using simple equality check as requested
    if row['n_sentences'] == 1:
        if prod_pattern.search(row['lyrics']):
            return np.nan
            
    return row['lyrics']

def detect_language(lyrics):
    #handle NaN, empty, or non-string lyrics
    if pd.isna(lyrics) or not isinstance(lyrics, str) or not lyrics.strip():
        return pd.NA

    try:
        return detect(lyrics)
    except LangDetectException:
        #handle cases with no detectable features (e.g., "!!!", "123")
        return pd.NA
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.NA

def find_swear_words_in_lyrics(lyrics, swear_words_list):
    if pd.isna(lyrics):
        return []

    lyrics_lower = lyrics.lower() #case-insensitive matching
    
    found_words = []
    for word in swear_words_list:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, lyrics_lower):
            found_words.append(word)
    
    return found_words

def count_swear_words_in_lyrics(lyrics, swear_words_list):
    if pd.isna(lyrics):
        return 0
    
    lyrics_lower = lyrics.lower()
    total_count = 0
    
    for word in swear_words_list:
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, lyrics_lower)
        total_count += len(matches)
    
    return total_count

def count_sentences_in_lyrics(lyrics):
    if pd.isna(lyrics):
        return 0
    lines = lyrics.split('\n')
    count = sum(1 for line in lines if line.strip())
    return count

def count_sentences_spacy(lyrics, lang, models_tokenizers, stored_n_sentences):
    if pd.isna(lyrics):
        return 0
    #only process italian and english (only two models used in the analysis)
    if lang not in ['it', 'en']:
        #return NA for missing language, otherwise keep stored value
        return 0 if pd.isna(lang) else stored_n_sentences
    
    nlp = models_tokenizers[lang]
    doc = nlp(lyrics)
    return len(list(doc.sents))

def count_tokens_split_punctuation(lyrics):
    if pd.isna(lyrics):
        return 0
    import re
    #split on word boundaries - keeps alphanumeric sequences separate from punctuation
    # \w+ -> word characters (letters, numbers); [^\w\s] ->  punctuation (not words and not separators);
    tokens = re.findall(r'\w+|[^\w\s]', lyrics, flags=re.UNICODE)
    return len(tokens)

def count_chars_no_whitespace(lyrics): #count characters excluding all whitespace
    if pd.isna(lyrics):
        return 0
    return len(re.sub(r'\s', '', lyrics))

def compute_lexical_density_regex(lyrics):
    if pd.isna(lyrics):
        return 0
    
    all_tokens = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", lyrics.lower(), flags=re.UNICODE) #finds both words and numbers
    
    all_words = all_tokens #filter out any tokens that are just numbers
    
    total_words = len(all_words)
    if total_words == 0:
        return 0
    
    unique_words = len(set(all_words)) #count of unique words
    
    return unique_words / total_words

def analyze_avg_token_per_clause(lyrics, lang, models_pos_taggers, stored_avg_token_per_clause):
    if pd.isna(lyrics):
        return 0
    
    if pd.isna(lang):
        return stored_avg_token_per_clause

    if lang not in ['it', 'en']:
        return stored_avg_token_per_clause
    
    nlp = models_pos_taggers[lang]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        doc = nlp(lyrics)
        
    total_non_punct_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
    num_predicates = len([t for t in doc if t.pos_ in ('VERB', 'AUX')])
    
    if num_predicates == 0:
        return 0
    else:
        return total_non_punct_tokens / num_predicates
    
def clean_title(title):
    return re.sub(r"[\(\[].*?[\)\]]", "", str(title)).strip()

def fetch_date_components(row, sp_client):
    #if release date is already present, return existing values
    if pd.notna(row['year']) and pd.notna(row['month']) and pd.notna(row['day']):
        return row['year'], row['month'], row['day']

    if sp_client is None:
        return np.nan, np.nan, np.nan

    query = f"artist:{row['name_artist']} track:{row['title']}"
    try:
        results = sp_client.search(q=query, type='track', limit=1)
        
        #fallback search with cleaned title
        if not results['tracks']['items']:
             clean_query = f"artist:{row['name_artist']} track:{clean_title(row['title'])}"
             results = sp_client.search(q=clean_query, type='track', limit=1)
             
        items = results['tracks']['items']
        
        if items:
            release_date = items[0]['album']['release_date']
            parts = release_date.split('-')
            
            y = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else np.nan
            d = int(parts[2]) if len(parts) > 2 else np.nan

            final_y = y if pd.notna(y) else row['year']
            final_m = m if pd.notna(m) else row['month']
            final_d = d if pd.notna(d) else row['day']
            
            return final_y, final_m, final_d
            
    except Exception:
        pass
    
    return row['year'], row['month'], row['day']

# --- main ---

def main():
    print("Starting data preprocessing script")

    CLIENT_ID = '6c362c1a59244409b7a2042986817a69'         
    CLIENT_SECRET = '10e9d0585552426281cdd579bf564b5a'

    input_file = '../dataset/tracks.csv'
    output_file = '../dataset/cleaned_tracks.csv'

    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    DetectorFactory.seed = 0
    models_tokenizers, models_pos_taggers = load_spacy_models()
    
    if models_tokenizers is None or models_pos_taggers is None:
        print("Error: Could not load spaCy models. Aborting script.")
        return

    try:
        df = pd.read_csv(input_file, sep=',')
        print(f"Successfully loaded '{input_file}'.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading '{input_file}': {e}")
        return

    # 1. strip string columns
    print("Cleaning string columns")
    cols_to_strip = [
        "album_name", "album_type", "album_image",
        "id_album", "id", "id_artist", "name_artist", "full_title",
        "title", "featured_artists", "primary_artist", "language", "album"
    ]

    zero_width_pattern = r"[\u200b\u200c\u200d\uFEFF]"

    nbsp_pattern = r"[\xa0]"

    for col in cols_to_strip:
        if col in df.columns:
            #strip only non empty rows
            non_null_mask = df[col].notna()
            
            df.loc[non_null_mask, col] = (
                df.loc[non_null_mask, col]
                .astype(str)
                .str.replace(zero_width_pattern, "", regex=True)
                .str.replace(nbsp_pattern, " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # 2. fix duplication error record (TR367132)
    print("Fixing duplication error for ID: TR367132")
    mask = (
        ((df["title"].str.lower() == "bugie") & (df["album_name"].str.lower() != "madame")) |
        ((df["title"].str.lower() == "sentimi") & (df["album_name"].str.lower() != "sentimi"))
    )
    df = df.loc[~mask].copy()
    df.loc[
        (df["title"].str.lower() == "bugie") &
        (df["album_name"].str.lower() == "madame"),
        "year"
    ] = 2021
    df.reset_index(drop=True, inplace=True)

    # 3. Fix duplicate track IDs
    print("Fixing duplicate track IDs")
    not_unique_ids = df['id'].value_counts()
    not_unique_ids = not_unique_ids[not_unique_ids > 1]
    not_unique_ids_list = not_unique_ids.index.tolist()
    
    if not_unique_ids_list:
        existing_numbers = (
            df['id']
            .dropna()
            .str.extract(r'TR(\d+)')[0]
            .dropna()
            .astype(int)
        )
        max_existing_num = existing_numbers.max()
        existing_ids = set(df['id'].dropna())
        
        for dup_id in not_unique_ids_list:
            dup_indices = df.index[df['id'] == dup_id].tolist()
            for i, idx in enumerate(dup_indices):
                if i == 0:
                    continue
                while True:
                    max_existing_num += 1
                    new_id = f"TR{max_existing_num}"
                    if new_id not in existing_ids:
                        break
                df.at[idx, 'id'] = new_id   
                existing_ids.add(new_id)
        print(f"Assigned new IDs to duplicates.")

    # 4. normalize title, full title and featured artists to have same apostrophe
    print("Normalizing apostrophe in title fields")
    for col in ['full_title', 'title', 'featured_artists']:
        if col in df.columns:
            df[col] = (df[col]
                .str.replace('\xa0', ' ', regex=False)
                .str.replace('\u2019', "'", regex=False)
                .str.replace('\u2018', "'", regex=False)
                .str.replace('\u201c', '"', regex=False)
                .str.replace('\u201d', '"', regex=False))

    # 5. clean lyrics
    print("Cleaning lyrics")
    rescue_list = [
        "Intro (Napolimanicomio)", "Dammi Ancora", "Cin Cin", "Tu Non Hai Mai",
        "Pronte A Tutto", "Non Dimentico Più", "Come Ti Senti?", "0 tempo e 0 vento",
        "Vita Vera - Story", "Meteoriti (english)", "24h Non Stop", "Dimmi Che Farai",
        "Se ti girassi"
    ]
    df['lyrics'] = df.apply(clean_contributor_lyrics, axis=1, rescue_list=rescue_list) 
    df['lyrics'] = df['lyrics'].apply(nullify_bad_lyrics)

    lyrics_before = df['lyrics'].copy()
    df['lyrics'] = df.apply(remove_production_only_lyrics, axis=1)

    deleted_mask = (lyrics_before.notna()) & (df['lyrics'].isna())
    deleted_count = deleted_mask.sum()

    # 6. adjusting lyrics-dependent columns
    print("Adjusting metrics for tracks with no lyrics...")
    count_cols = ['n_tokens', 'n_sentences', 'swear_IT', 'swear_EN']
    ratio_cols = ['tokens_per_sent', 'char_per_tok', 'avg_token_per_clause', 'lexical_density']
    string_cols = ['language', 'swear_IT_words', 'swear_EN_words']
    missing_lyrics_mask = df['lyrics'].isna()
    
    df.loc[missing_lyrics_mask, count_cols] = 0
    df.loc[missing_lyrics_mask, ratio_cols] = 0.0
    for col in string_cols:
        df.loc[missing_lyrics_mask, col] = pd.NA

    # 7. fix language
    print("Fixing 'language' column")
    mask_to_check = df['language'].isna() & df['lyrics'].notna()
    if mask_to_check.sum() > 0:
        detected_languages = df.loc[mask_to_check, 'lyrics'].apply(detect_language)
        df.loc[detected_languages.index, 'language'] = detected_languages

    incorrect_italian_codes = [
        'pl', 'da', 'cs', 'nl', 'sr', 'war', 'eu', 'no', 'ia', 'ca', 
        'gl', 'sco', 'la', 'eo', 'rm', 'et', 'lt', 'aa', 'ro', 'rw', 
        'chr', 'qu', 'mt', 'cy', 'sq', 'sw', 'co'
    ]
    df.loc[df['language'].isin(incorrect_italian_codes), 'language'] = 'it'

    mixed_languages = ['en', 'es', 'pt', 'fr']
    mask_to_recheck = df['language'].isin(mixed_languages) & df['lyrics'].notna()
    if mask_to_recheck.sum() > 0:
        detected_languages_mixed = df.loc[mask_to_recheck, 'lyrics'].apply(detect_language)
        update_mask = detected_languages_mixed.notna()
        df.loc[detected_languages_mixed[update_mask].index, 'language'] = detected_languages_mixed[update_mask].values

    manual_updates = {
        "KI-KI": "es",
        "The Banana Splits": "en",
        "Intro (Monkee Bizniz Vol. 2)": "en",
        "FOXY": "en"
    }
    for title, lang in manual_updates.items():
        df.loc[df['title'] == title, 'language'] = lang
        
    # replace string 'nan' with proper NaN
    df['language'] = df['language'].replace('nan', pd.NA)

    # 8. Fix duplicate album IDs
    print("Fixing duplicate album IDs")

    album_name_per_id = df.groupby('id_album')['album_name'].nunique()
    problematic_album_ids = album_name_per_id[album_name_per_id > 1].index.tolist()

    existing_numbers = (
        df['id_album']
        .dropna()
        .str.extract(r'ALB(\d+)')[0]
        .dropna()
        .astype(int)
    )
    max_existing_num = existing_numbers.max()
    existing_ids = set(df['id_album'].dropna())
    
    new_ids_assigned = 0
    for dup_album_id in problematic_album_ids:
        dup_indices = df.index[df['id_album'] == dup_album_id].tolist()
        
        album_groups = df.loc[dup_indices].groupby('album_name').groups
        
        #keep the first group unchanged, assign new IDs to others
        first_group = True
        for album_name, indices in album_groups.items():
            if first_group:
                first_group = False
                continue
            
            while True:
                max_existing_num += 1
                new_album_id = f"ALB{max_existing_num:06d}"
                if new_album_id not in existing_ids:
                    break
            
            for idx in indices:
                df.at[idx, 'id_album'] = new_album_id
                new_ids_assigned += 1
            
            existing_ids.add(new_album_id)
    print(f"Fixed album IDs for {new_ids_assigned} rows.")

    #shared album names (unreliable for imputation)
    albums_per_name = df.groupby('album_name')['id_album'].nunique()
    shared_album_names = albums_per_name[albums_per_name > 1].index.tolist()

    # 9. standardize different album release dates
    print("Standardizing album release dates.")

    df['album_release_date'] = pd.to_datetime(df['album_release_date'], errors='coerce')
    
    dates_per_id = df.groupby('id_album')['album_release_date'].nunique()
    problematic_date_ids = dates_per_id[dates_per_id > 1].index.tolist()
    
    date_fixes = 0
    for aid in problematic_date_ids:
        current_names = df.loc[df['id_album'] == aid, 'album_name'].dropna().unique()
        if any(name in shared_album_names for name in current_names):
            continue

        #use the earliest date for this album ID
        earliest_date = df[df['id_album'] == aid]['album_release_date'].min()
        
        if pd.notna(earliest_date):
            mask = df['id_album'] == aid
            df.loc[mask, 'album_release_date'] = earliest_date
            date_fixes += mask.sum()
    print(f"Standardized release dates for {date_fixes} rows.")

    # 10. fix numeric types
    print("fixing numeric columns")
    df['stats_pageviews'] = pd.to_numeric(df['stats_pageviews'], errors='coerce').astype('Int64')
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
    df['day'] = pd.to_numeric(df['day'], errors='coerce').astype('Int64')

    df.loc[(df['year'] < 1973) | (df['year'] > 2025), 'year'] = pd.NA

    initial_missing = df['year'].isna().sum()

    print("Fetching missing date components from Spotify API...")
    initial_missing = df['year'].isna().sum()
    initial_missing_dates = df[['year', 'month', 'day']].isna().any(axis=1).sum()
    
    date_components = df.apply(lambda row: fetch_date_components(row, sp), axis=1, result_type='expand')
    df[['year', 'month', 'day']] = date_components
    
    # Ensure correct types after update
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
    df['day'] = pd.to_numeric(df['day'], errors='coerce').astype('Int64')
    
    final_missing = df['year'].isna().sum()
    final_missing_dates = df[['year', 'month', 'day']].isna().any(axis=1).sum()
    print(f"Filled {initial_missing - final_missing} years using Spotify API.")
    print(f"Filled {initial_missing_dates - final_missing_dates} rows using Spotify API.")

    print("Filling remaining missing year/month/day (using reliable albums only)")
    #mask where release date exists AND album name is not shared
    mask_date_available = df['album_release_date'].notna()
    mask_reliable_album = ~df['album_name'].isin(shared_album_names)
    mask_imputation = mask_date_available & mask_reliable_album

    #fill year
    mask_fill_year = df['year'].isna() & mask_imputation
    df.loc[mask_fill_year, 'year'] = df.loc[mask_fill_year, 'album_release_date'].dt.year
    df['year'] = df['year'].astype('Int64')

    #fill month
    mask_fill_month = df['month'].isna() & mask_imputation
    df.loc[mask_fill_month, 'month'] = df.loc[mask_fill_month, 'album_release_date'].dt.month
    df['month'] = df['month'].astype('Int64')

    #fill day
    mask_fill_day = df['day'].isna() & mask_imputation
    df.loc[mask_fill_day, 'day'] = df.loc[mask_fill_day, 'album_release_date'].dt.day
    df['day'] = df['day'].astype('Int64')
    
    # 11. recalculate swear words
    print("Recalculating swear words...")
    it_words_to_exclude = ['water', 'blowjob', 'jug', 'toro', 'fortuna', 'zanzara', 'granchio']
    it_words_to_include = [
        'bastarda', 'battone', 'bocchini', 'cagate', 'cagne', 'cazzate', 'cazzoni', 'cessa', 'checche', 
        'chiappe', 'fessi', 'fiche', 'fighe', 'finocchi', 'fregne', 'froci', 'incazzati', 'merde', 
        'mignotte', 'minchiate', 'pompini', 'porche', 'puttane', 'ricchioni', 'scopate', 'seghe', 'sorche',
        'stronzate', 'stronzi', 'stronza', 'stronze', 'stupide', 'stupidi', 'tette', 'troie', 'zoccole'
    ]
    base_it_swears = {
        'arrapante', 'arrapare', 'arrapato', 'bagascia', 'bastardi', 'bastardo', 'battona', 'bernarda', 
        'bischero', 'blowjob', 'bocchinaro', 'bocchino', 'bombare', 'cacare', 'cacata', 'cacca', 'cagare', 
        'cagata', 'cagna', 'cappella', 'cazzata', 'cazzeggiare', 'cazzeggio', 'cazzi', 'cazzo', 'cazzone', 
        'cazzuto', 'cesso', 'checca', 'chiappa', 'chiavare', 'chiavata', 'cogliona', 'coglionata', 'coglione', 
        'coglioni', 'controcazzi', 'controcoglioni', 'cornuto', 'cozza', 'cretina', 'cretini', 'cretino', 
        'culattone', 'culo', 'cunnu', 'deretano', 'escremento', 'farabutti', 'farabutto', 'fava', 'feci', 
        'fellatio', 'fesso', 'fica', 'fico', 'figa', 'figo', 'finocchio', 'fogna', 'fogne', 'fortuna', 
        'fottere', 'fottersi', 'fottio', 'fottuti', 'fottuto', 'fregare', 'fregarsene', 'fregna', 'frocio', 
        'gay', 'gigolo', 'glutei', 'gnocca', 'granchio', 'grilletto', 'handicappato', 'idiozia', 'incazzare', 
        'incazzarsi', 'incazzato', 'inculare', 'jug', 'leccaculo', 'madonna', 'maiala', 'maroni', 'mazzo', 
        'merda', 'merdaio', 'merdaiolo', 'merdata', 'merdina', 'mezzasega', 'mignotta', 'minchiata', 
        'minchioni', 'missionario', 'nerchia', 'palle', 'paraculo', 'pecorina', 'peluria', 'pene', 'piccione', 
        'pipa', 'pippa', 'pisciare', 'pisciata', 'piscio', 'pisello', 'pompinara', 'pompino', 'porca', 
        'pugnetta', 'puttana', 'puttanaio', 'puttanata', 'puttaniere', 'puttano', 'raspa', 'ricchione', 
        'rompicoglioni', 'rompipalle', 'rottinculo', 'sboccare', 'sborra', 'sborrare', 'scassare', 'scazzato', 
        'scazzo', 'schizzare', 'scopare', 'scopata', 'scoreggia', 'scrofa', 'seccatore', 'sedere', 'sega', 
        'segaiolo', 'selvaggio', 'sfiga', 'sfigata', 'sgualdrina', 'smerdare', 'sorca', 'spagnola', 
        'spompinare', 'sputtanare', 'strafottenza', 'stronzata', 'stronzo', 'stupida', 'stupido', 'sveltina', 
        'tetta', 'topa', 'toro', 'travestito', 'troia', 'troiaggine', 'troiaio', 'trombare', 'uccello', 
        'vacca', 'vaccata', 'vaffanculo', 'vagina', 'water', 'zanzara', 'zizza', 'zoccola'
    }
    filtered_it_swear_words = {word for word in base_it_swears if word not in it_words_to_exclude}
    filtered_it_swear_words.update(it_words_to_include)

    base_en_swears = {
        'snatch', 'horny', 'fag', 'bitches', 'topless', 'rapist', 'tranny', 'bondage', 'scat', 'xx', 
        'doggystyle', 'cialis', 'hooker', 'fucking', 'porno', 'cum', 'cumming', 'ass', 'poof', 'dick', 
        'semen', 'anal', 'clit', 'cocks', 'vagina', 'rape', 'nude', 'titty', 'pussy', 'spic', 'coon', 
        'milf', 'pissing', 'fellatio', 'cunt', 'xxx', 'creampie', 'rimming', 'lolita', 'negro', 'sexy', 
        'slut', 'shit', 'faggot', 'playboy', 'pedobear', 'vulva', 'viagra', 'panties', 'cumshot', 
        'skeet', 'gangbang', 'bbw', 'tit', 'domination', 'fisting', 'porn', 'bullshit', 'boobs', 'butt', 
        'sexual', 'busty', 'kinky', 'tits', 'asshole', 'sucks', 'fuckin', 'cock', 'hardcore', 'shibari', 
        'punany', 'masturbation', 'ecchi', 'blowjob', 'bastard', 'suck', 'raping', 'motherfucker', 
        'voyeur', 'sex', 'dildo', 'hentai', 'sexo', 'neonazi', 'anus', 'shitty', 'threesome', 'bukkake', 
        'bastardo', 'escort', 'deepthroat', 'nympho', 'bitch', 'nipple', 'nigga', 'fuck'
    }
    en_words_to_exclude = ['bastardo']
    filtered_en_swear_words = {word for word in base_en_swears if word not in en_words_to_exclude}
    
    # update swear word lists
    df['swear_IT_words'] = df['lyrics'].apply(lambda x: str(find_swear_words_in_lyrics(x, filtered_it_swear_words)))
    df['swear_EN_words'] = df['lyrics'].apply(lambda x: str(find_swear_words_in_lyrics(x, filtered_en_swear_words)))
    
    # update swear word counts
    df['swear_IT'] = df.apply(lambda row: count_swear_words_in_lyrics(row['lyrics'], ast.literal_eval(row['swear_IT_words']) if pd.notna(row['swear_IT_words']) else []), axis=1)
    df['swear_EN'] = df.apply(lambda row: count_swear_words_in_lyrics(row['lyrics'], ast.literal_eval(row['swear_EN_words']) if pd.notna(row['swear_EN_words']) else []), axis=1)

    # 12. recalculate/fill lyric metrics
    print("Recalculating and filling missing lyric metrics...")

    # use computed_lexical_density (Regex) as new standard
    df['lexical_density'] = df['lyrics'].apply(compute_lexical_density_regex)
    
    # n_sentences
    df['n_sentences'] = pd.to_numeric(df['n_sentences'], errors='coerce').astype('Int64')
    missing_n_sentences = ((df['n_sentences'].isna()) | (df['n_sentences'] == 0)) & df['lyrics'].notna()
    for idx in df[missing_n_sentences].index:
        lyrics = df.at[idx, 'lyrics']
        non_empty_count = count_sentences_in_lyrics(lyrics)
        if non_empty_count == 1:
            lang = df.at[idx, 'language']
            spacy_count = count_sentences_spacy(lyrics, lang, models_tokenizers, pd.NA)
            df.at[idx, 'n_sentences'] = spacy_count
        else:
            df.at[idx, 'n_sentences'] = non_empty_count

    # n_tokens
    df['n_tokens'] = pd.to_numeric(df['n_tokens'], errors='coerce').astype('Int64')
    missing_n_tokens = ((df['n_tokens'].isna()) | (df['n_tokens'] == 0)) & df['lyrics'].notna()
    for idx in df[missing_n_tokens].index:
        df.at[idx, 'n_tokens'] = count_tokens_split_punctuation(df.at[idx, 'lyrics'])

    # tokens_per_sent
    mask_fill_tps = ((df['tokens_per_sent'].isna()) | (df['tokens_per_sent'] == 0)) & df['n_tokens'].notna() & df['n_sentences'].notna() & (df['n_sentences'] != 0)
    df.loc[mask_fill_tps, 'tokens_per_sent'] = df.loc[mask_fill_tps, 'n_tokens'] / df.loc[mask_fill_tps, 'n_sentences']

    # char_per_tok
    mask_fill_cpt = ((df['char_per_tok'].isna()) | (df['char_per_tok'] == 0)) & df['n_tokens'].notna() & (df['n_tokens'] != 0)
    indices_fill_cpt = df[mask_fill_cpt].index
    if not indices_fill_cpt.empty:
        chars_no_ws = df.loc[indices_fill_cpt, 'lyrics'].apply(count_chars_no_whitespace)
        n_tokens = df.loc[indices_fill_cpt, 'n_tokens']
        df.loc[indices_fill_cpt, 'char_per_tok'] = chars_no_ws / n_tokens

    # lexical_density
    computed_ld_regex = df['lyrics'].apply(compute_lexical_density_regex)
    mask_fill_ld = ((df['lexical_density'].isna()) | (df['lexical_density'] == 0)) & computed_ld_regex.notna()
    df.loc[mask_fill_ld, 'lexical_density'] = computed_ld_regex[mask_fill_ld]

    # avg_token_per_clause
    cmp_avg_token_per_clause = df.apply(
        lambda row: analyze_avg_token_per_clause(row['lyrics'], row['language'], models_pos_taggers, row['avg_token_per_clause']),
        axis=1
    )
    mask_fill_atpc = ((df['avg_token_per_clause'].isna()) | (df['avg_token_per_clause'] == 0)) & cmp_avg_token_per_clause.notna() & (cmp_avg_token_per_clause != 0)
    df.loc[mask_fill_atpc, 'avg_token_per_clause'] = cmp_avg_token_per_clause[mask_fill_atpc]

    #handle cases where tokens_per_sent is high with only one sentence = recompute number of sentences
    target_mask = (df['tokens_per_sent'] > 15) & (df['n_sentences'] == 1) & (df['lyrics'].notna())
    
    if target_mask.sum() > 0:
        for idx in df[target_mask].index:
            lyrics = df.at[idx, 'lyrics']
            non_empty_count = count_sentences_in_lyrics(lyrics)
            
            if non_empty_count == 1:
                lang = df.at[idx, 'language']
                spacy_count = count_sentences_spacy(lyrics, lang, models_tokenizers, pd.NA)
                df.at[idx, 'n_sentences'] = spacy_count
            else:
                df.at[idx, 'n_sentences'] = non_empty_count
        
        # recalculate tokens_per_sent for these rows
        df.loc[target_mask, 'n_tokens'] = df.loc[target_mask, 'lyrics'].apply(count_tokens_split_punctuation)
        df.loc[target_mask, 'tokens_per_sent'] = df.loc[target_mask, 'n_tokens'] / df.loc[target_mask, 'n_sentences']

    print("Refining avg_token_per_clause for outliers...")
    threshold_atpc = df['avg_token_per_clause'].quantile(0.98)
    outliers_mask = (df['avg_token_per_clause'] > threshold_atpc) & (df['lyrics'].notna())
    
    if outliers_mask.sum() > 0:
        new_atpc = df.loc[outliers_mask].apply(
            lambda row: analyze_avg_token_per_clause(
                row['lyrics'], 
                row['language'], 
                models_pos_taggers, 
                row['avg_token_per_clause']
            ), axis=1
        )
        df.loc[outliers_mask, 'avg_token_per_clause'] = new_atpc

    target_track_mask = df['title'].str.contains("Ne se obrushtai", na=False, regex=False) | \
                        df['title'].str.contains("Не се обръщай", na=False, regex=False)
    if target_track_mask.any():
        df.loc[target_track_mask, 'avg_token_per_clause'] = 3.95
    target_track_mask = df['title'].str.contains("Бары", na=False, regex=False)
    if target_track_mask.any():
        df.loc[target_track_mask, 'avg_token_per_clause'] = 5.16

    # 13. fix audio feature outliers
    print("Fixing audio feature outliers using thresholds...")

    mask_bpm = df['bpm'] > 250
    print(f"  - Removing {mask_bpm.sum()} BPM outliers (>250)")
    df.loc[mask_bpm, 'bpm'] = np.nan
    
    mask_centroid = df['centroid'] < 0.001
    print(f"  - Removing {mask_centroid.sum()} Centroid outliers (<0.001)")
    df.loc[mask_centroid, 'centroid'] = np.nan
    
    mask_rolloff = df['rolloff'] < 20.0
    print(f"  - Removing {mask_rolloff.sum()} Rolloff outliers (<20.0)")
    df.loc[mask_rolloff, 'rolloff'] = np.nan
    
    mask_flux = df['flux'] < 0.001
    print(f"  - Removing {mask_flux.sum()} Flux outliers (<0.001)")
    df.loc[mask_flux, 'flux'] = np.nan
    
    mask_loudness = df['loudness'] < 0.001
    print(f"  - Removing {mask_loudness.sum()} Loudness outliers (<0.001)")
    df.loc[mask_loudness, 'loudness'] = np.nan
    # 14. fix popularity
    print("Fixing 'popularity' column...")
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    invalid_range_mask = (df['popularity'] < 0) | (df['popularity'] > 100)
    df.loc[invalid_range_mask, 'popularity'] = np.nan
    df['popularity'] = df['popularity'].astype('Int64')

    # 15. fix explicit
    print("Recalculating 'explicit' column based on swear words...")
    df['total_swear_words'] = df['swear_IT'] + df['swear_EN']
    df.loc[(df['total_swear_words'] == 0), 'explicit'] = False
    df.loc[(df['total_swear_words'] > 0), 'explicit'] = True
    df['explicit'] = df['explicit'].astype('boolean') #use nullable boolean

    # 16. drop helper columns
    cols_to_drop = ['total_swear_words']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 17. save cleaned data
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved preprocessed data to '{output_file}'.")
    except Exception as e:
        print(f"Error saving file to '{output_file}': {e}")

if __name__ == "__main__":
    main()
