import pandas as pd
import numpy as np
import os
import re
import requests
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

DATASET_DIR = os.path.join('..', 'dataset')
INPUT_FILE = os.path.join(DATASET_DIR, 'artists.csv')
TRACKS_FILE = os.path.join(DATASET_DIR, 'cleaned_tracks.csv')
OUTPUT_FILE = os.path.join(DATASET_DIR, 'cleaned_artists.csv')

USER_AGENT = "artist_data_cleaning_script_v1"

def clean_string_columns(df):
    cols_to_strip = [
        "id_author", "name", "gender", "birth_date",
        "birth_place", "nationality", "description",
        "active_start", "province", "region", "country"
    ]
    
    zero_width_pattern = r"[\u200b\u200c\u200d\uFEFF]"
    nbsp_pattern = r"[\xa0]"
    
    for col in cols_to_strip:
        if col in df.columns:
            non_null_mask = df[col].notna()
            df.loc[non_null_mask, col] = (
                df.loc[non_null_mask, col]
                .astype(str)
                .str.replace(zero_width_pattern, "", regex=True)
                .str.replace(nbsp_pattern, " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return df

def process_dates(df):
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    
    df['active_start'] = pd.to_datetime(df['active_start'], errors='coerce')
    
    return df

def enrich_active_start_from_tracks(df, tracks_path):
    if not os.path.exists(tracks_path):
        print(f"Warning: Tracks file not found at {tracks_path}. Skipping enrichment.")
        return df

    df_tracks = pd.read_csv(tracks_path)
    
    #filter valid years
    valid_tracks = df_tracks[df_tracks['year'].notna()].copy()
    
    #fill default month/day for construction
    valid_tracks['month'] = valid_tracks['month'].fillna(1).astype(int)
    valid_tracks['day'] = valid_tracks['day'].fillna(1).astype(int)
    valid_tracks['year'] = valid_tracks['year'].astype(int)
    
    valid_tracks['release_date_constructed'] = pd.to_datetime(
        valid_tracks[['year', 'month', 'day']], errors='coerce'
    )
    
    #find earliest date per artist
    earliest_dates = (
        valid_tracks.groupby('id_artist')['release_date_constructed']
        .min()
        .reset_index()
        .rename(columns={'release_date_constructed': 'first_track_release'})
    )
    
    #merge and Fill
    df = pd.merge(df, earliest_dates, left_on='id_author', right_on='id_artist', how='left')
    
    missing_active = df['active_start'].isna()
    recoverable = missing_active & df['first_track_release'].notna()
    
    if recoverable.sum() > 0:
        print(f"Recovered {recoverable.sum()} active_start dates.")
        df.loc[recoverable, 'active_start'] = df.loc[recoverable, 'first_track_release']
    
    df.drop(columns=['first_track_release'], inplace=True)
    return df

def get_osm_address_from_coords(row, geolocator):
    try:
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            query = f"{row['latitude']}, {row['longitude']}"
            location = geolocator(query, language='it')
            if location and location.raw.get('address'):
                addr = location.raw['address']
                found_region = addr.get('state')
                #fallback: county -> city
                found_province = addr.get('county', addr.get('city')) 
                return pd.Series([found_province, found_region])
    except Exception:
        pass
    return pd.Series([None, None])

def get_wikidata_id(name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'search': name,
        'language': 'it',
        'format': 'json',
        'limit': 1
    }
    headers = {'User-Agent': 'ArtistCleaningScript/1.0'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        if data.get('success') and data.get('search'):
            return data['search'][0]['id']
    except Exception:
        return None
    return None

def get_location_details_wikidata(qid):
    if not qid: 
        return None, None
        
    url = "https://query.wikidata.org/sparql"
    # Query: find birth place (P19) or formation (P740), traverse admin unit (P131)
    query = f"""
    SELECT DISTINCT ?label ?typeLabel WHERE {{
      VALUES ?artist {{ wd:{qid} }}      
      {{ ?artist wdt:P19 ?place. }} UNION {{ ?artist wdt:P740 ?place. }}      
      ?place wdt:P131* ?admin.
      ?admin wdt:P31 ?type.
      VALUES ?type {{ wd:Q16110 wd:Q15089 wd:Q15110 }}     
      ?admin rdfs:label ?label.
      FILTER(LANG(?label) = "it").
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "it". }}
    }} LIMIT 2
    """
    headers = {'User-Agent': 'ArtistCleaningScript/1.0'}
    try:
        r = requests.get(url, params={'format': 'json', 'query': query}, headers=headers, timeout=5)
        data = r.json()
        
        region, province = None, None
        known_regions = [
            "Lombardia", "Lazio", "Campania", "Sicilia", "Veneto", "Piemonte", 
            "Emilia-Romagna", "Puglia", "Toscana", "Calabria", "Sardegna", "Liguria", 
            "Marche", "Abruzzo", "Friuli-Venezia Giulia", "Trentino-Alto Adige", 
            "Umbria", "Basilicata", "Molise", "Valle d'Aosta"
        ]
        
        for item in data['results']['bindings']:
            label = item['label']['value']
            clean_label = (label.replace("città metropolitana di ", "")
                               .replace("Città metropolitana di ", "")
                               .replace("provincia di ", "")
                               .replace("Provincia di ", "").strip())
            
            if clean_label in known_regions:
                region = clean_label
            else:
                province = clean_label
        
        return province, region
    except Exception:
        return None, None

def fetch_artist_geo_data(name):
    qid = get_wikidata_id(name)
    return get_location_details_wikidata(qid)

def fix_artist_geography_manual(df):
    
    verified_geo_data = {
        'beba': {'province': 'Torino', 'region': 'Piemonte'},
        'bigmama': {'province': 'Avellino', 'region': 'Campania'},
        'brusco': {'province': 'Roma', 'region': 'Lazio'},
        'bushwaka': {'province': 'La Spezia', 'region': 'Liguria'},
        'caneda': {'province': 'Milano', 'region': 'Lombardia'},
        'colle der fomento': {'province': 'Roma', 'region': 'Lazio'},
        'cor veleno': {'province': 'Roma', 'region': 'Lazio'},
        'dark polo gang': {'province': 'Roma', 'region': 'Lazio'},
        'doll kill': {'province': 'Sassari', 'region': 'Sardegna'},
        'eva rea': {'province': 'Catania', 'region': 'Sicilia'},
        'hindaco': {'province': 'Milano', 'region': 'Lombardia'},
        'joey funboy': {'province': 'Bolzano', 'region': 'Trentino-Alto Adige'},
        'johnny marsiglia': {'province': 'Palermo', 'region': 'Sicilia'},
        'miss simpatia': {'province': 'Ancona', 'region': 'Marche'},
        'mistico': {'province': 'Milano', 'region': 'Lombardia'},
        'priestess': {'province': 'Bari', 'region': 'Puglia'},
        'samuel heron': {'province': 'La Spezia', 'region': 'Liguria'},
        'shiva': {'province': 'Milano', 'region': 'Lombardia'},
        'skioffi': {'province': 'Frosinone', 'region': 'Lazio'},
        'sottotono': {'province': 'Milano', 'region': 'Lombardia'},
        'yendry': {'province': 'Torino', 'region': 'Piemonte'}
    }

    for index, row in df.iterrows():
        artist_key = str(row['name']).lower().strip()
        #handle specific edge case
        if 'ye' in artist_key and 'dry' in artist_key: 
             artist_key = 'yendry'

        if artist_key in verified_geo_data:
            data = verified_geo_data[artist_key]
            df.at[index, 'province'] = data['province']
            df.at[index, 'region'] = data['region']
            
    return df

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE, sep=';')

    # 1. clean strings
    df = clean_string_columns(df)

    # 2. process dates
    df = process_dates(df)
    
    # 3. enrich active start from earliest tracks
    df = enrich_active_start_from_tracks(df, TRACKS_FILE)

    # 4. drop useless columns (based on analysis)
    print("Dropping irrelevant columns (active_end, description, country)...")
    df.drop(columns=['active_end', 'description', 'country'], inplace=True, errors='ignore')

    # 5. geographic enrichment
    print("Starting Geographic Enrichment (this may take time)...")
    
    osm_locator = Nominatim(user_agent=USER_AGENT, timeout=10)
    reverse_osm = RateLimiter(osm_locator.reverse, min_delay_seconds=1.0)
    
    mask_has_coords = (df['latitude'].notna()) & (df['longitude'].notna()) & (df['province'].isna())
    if mask_has_coords.sum() > 0:
        print(f"Reverse geocoding {mask_has_coords.sum()} rows...")
        tqdm.pandas(desc="Reverse Geocoding")
        df.loc[mask_has_coords, ['province', 'region']] = df[mask_has_coords].progress_apply(
            lambda row: get_osm_address_from_coords(row, reverse_osm), axis=1
        ).values

    #Wikidata fetching for missing data
    missing_loc_mask = df['province'].isna() | df['region'].isna()
    
    if missing_loc_mask.sum() > 0:
        print(f"Fetching Wikidata for {missing_loc_mask.sum()} artists with missing location...")
        
        rows_to_fetch = df[missing_loc_mask].copy()
        
        if os.path.exists(TRACKS_FILE):
            df_tracks = pd.read_csv(TRACKS_FILE)
            track_names = df_tracks[['id_artist', 'name_artist']].drop_duplicates().rename(columns={'id_artist': 'id_author'})
            rows_to_fetch = pd.merge(rows_to_fetch, track_names, on='id_author', how='left')
        else:
            rows_to_fetch['name_artist'] = np.nan

        def fetch_wrapper(row):
            prov, reg = fetch_artist_geo_data(row['name'])
            
            if (not prov and not reg) and pd.notna(row.get('name_artist')):
                if str(row['name']).lower() != str(row['name_artist']).lower():
                    prov, reg = fetch_artist_geo_data(row['name_artist'])
            
            return pd.Series([prov, reg])

        tqdm.pandas(desc="Wikidata Fetch")
        fetched_data = rows_to_fetch.progress_apply(fetch_wrapper, axis=1)
        
        df.loc[missing_loc_mask, 'province'] = df.loc[missing_loc_mask, 'province'].fillna(fetched_data[0])
        df.loc[missing_loc_mask, 'region'] = df.loc[missing_loc_mask, 'region'].fillna(fetched_data[1])

    mask_missing_region = df['region'].isna() & df['province'].notna()
    if mask_missing_region.sum() > 0:
        print(f"Inferring Region for {mask_missing_region.sum()} rows based on Province...")
        
        geocode_osm = RateLimiter(osm_locator.geocode, min_delay_seconds=1.0)
        
        provinces = df.loc[mask_missing_region, 'province'].unique()
        prov_map = {}
        
        for prov in tqdm(provinces, desc="Province Lookup"):
            try:
                loc = geocode_osm(f"{prov}, Italy", addressdetails=True, language='it')
                if loc and 'address' in loc.raw:
                    prov_map[prov] = loc.raw['address'].get('state')
            except:
                continue
        
        df.loc[mask_missing_region, 'region'] = df.loc[mask_missing_region, 'province'].map(prov_map)

    # 6. Manual Fixes (Highest Priority Override)
    df = fix_artist_geography_manual(df)

    # 7. Save
    print(f"Saving cleaned data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, sep=';')

if __name__ == "__main__":
    main()