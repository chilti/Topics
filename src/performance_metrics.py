import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import pyarrow.parquet as pq

def safe_get(obj, *keys, default=None):
    """Safely navigate nested dictionaries."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        else:
            return default
    return obj if obj is not None else default

def parse_json_field(value):
    """Parse JSON string to dict/list if needed."""
    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
        try:
            return json.loads(value)
        except:
            return value
    return value

def get_cache_dir():
    """Returns the cache directory path."""
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

class MetricsAccumulator:
    """
    Accumulates metrics across chunks for memory-efficient processing.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.count = 0
        self.fwci_sum = 0.0
        self.percentile_sum = 0.0
        self.top_10_count = 0
        self.top_1_count = 0
        self.oa_counts = {
            'gold': 0,
            'diamond': 0,
            'green': 0,
            'hybrid': 0,
            'bronze': 0,
            'closed': 0
        }
        # Language counters
        self.lang_counts = {
            'en': 0, # English
            'fr': 0, # French
            'de': 0, # German
            'it': 0, # Italian
            'la': 0, # Latin
            'nd': 0, # North Ndebele
            'pt': 0, # Portuguese
            'ru': 0, # Russian
            'es': 0, # Spanish
            'other': 0
        }

    def add_batch(self, chunk):
        """
        Process a DataFrame chunk and update metrics.
        chunk is a DataFrame with works data.
        """
        if len(chunk) == 0:
            return

        self.count += len(chunk)
        
        # FWCI (Field Weighted Citation Impact)
        if 'fwci' in chunk.columns:
            # Replace None/NaN with 0 or exclude? Usually 1.0 is neutral, but 0.0 if unknown.
            # Assuming pre-filled 0.0 for unknown
            self.fwci_sum += chunk['fwci'].fillna(0.0).sum()
        elif 'cited_by_count' in chunk.columns:
            # Rough proxy if FWCI missing (not ideal but better than nothing)
            # Normalizing by something would be better, but we sum raw counts here? 
            # No, variable name is fwci_sum. If missing, we add 0.
            pass

        # Percentile
        if 'citation_normalized_percentile' in chunk.columns:
            # Usually stored as decimal 0-100 or 0-1 in data? 
            # OpenAlex: citation_normalized_percentile.value is 0-100.
            # We assume the column contains the numeric value.
            
            # Handle potential dictionary column if not unpacked
            if chunk['citation_normalized_percentile'].dtype == 'object':
                 # Try to extract value if it's a dict/json
                 vals = pd.to_numeric(chunk['citation_normalized_percentile'], errors='coerce').fillna(0.0)
            else:
                 vals = chunk['citation_normalized_percentile'].fillna(0.0)
            
            self.percentile_sum += vals.sum()
            
            # Top 10% (>= 90.0)
            self.top_10_count += (vals >= 90.0).sum()
            
            # Top 1% (>= 99.0)
            self.top_1_count += (vals >= 99.0).sum()
            
        # Open Access Status
        if 'oa_status' in chunk.columns:
            counts = chunk['oa_status'].value_counts()
            for status in ['gold', 'diamond', 'green', 'hybrid', 'bronze', 'closed']:
                self.oa_counts[status] += counts.get(status, 0)
                
        # Language Stats
        if 'language' in chunk.columns:
            # Count languages
            lang_counts = chunk['language'].fillna('unknown').value_counts()
            
            target_langs = ['en', 'fr', 'de', 'it', 'la', 'nd', 'pt', 'ru', 'es']
            
            for lang in target_langs:
                self.lang_counts[lang] += lang_counts.get(lang, 0)
                
            # Count others (all that are not in our target list)
            # sum of counts for indices NOT in target_langs
            self.lang_counts['other'] += lang_counts[~lang_counts.index.isin(target_langs)].sum()

    def get_metrics(self):
        """
        Return dictionary with calculated average metrics.
        """
        if self.count == 0:
            return {
                'num_documents': 0,
                'fwci_avg': 0.0,
                'pct_top_10': 0.0,
                'pct_top_1': 0.0,
                'avg_percentile': 0.0,
                'pct_oa_gold': 0.0,
                'pct_oa_diamond': 0.0,
                'pct_oa_green': 0.0,
                'pct_oa_hybrid': 0.0,
                'pct_oa_bronze': 0.0,
                'pct_oa_closed': 0.0,
                
                # Default languages
                'pct_lang_en': 0.0,
                'pct_lang_fr': 0.0,
                'pct_lang_de': 0.0,
                'pct_lang_it': 0.0,
                'pct_lang_la': 0.0,
                'pct_lang_nd': 0.0,
                'pct_lang_pt': 0.0,
                'pct_lang_ru': 0.0,
                'pct_lang_es': 0.0,
                'pct_lang_other': 0.0
            }
        
        metrics = {
            'num_documents': self.count,
            'fwci_avg': round(self.fwci_sum / self.count, 2),
            'pct_top_10': round((self.top_10_count / self.count) * 100, 2),
            'pct_top_1': round((self.top_1_count / self.count) * 100, 2),
            'avg_percentile': round(self.percentile_sum / self.count, 2),
            'pct_oa_gold': round((self.oa_counts['gold'] / self.count) * 100, 2),
            'pct_oa_diamond': round((self.oa_counts['diamond'] / self.count) * 100, 2),
            'pct_oa_green': round((self.oa_counts['green'] / self.count) * 100, 2),
            'pct_oa_hybrid': round((self.oa_counts['hybrid'] / self.count) * 100, 2),
            'pct_oa_bronze': round((self.oa_counts['bronze'] / self.count) * 100, 2),
            'pct_oa_closed': round((self.oa_counts['closed'] / self.count) * 100, 2)
        }
        
        # Calculate Language Percentages
        for lang in ['en', 'fr', 'de', 'it', 'la', 'nd', 'pt', 'ru', 'es', 'other']:
            metrics[f'pct_lang_{lang}'] = round((self.lang_counts[lang] / self.count) * 100, 2)
            
        return metrics

def process_works_in_chunks(works_filepath, filter_func=None, chunk_size=50000):
    """
    Process works file in chunks and calculate metrics.
    
    Args:
        works_filepath: Path to works parquet file
        filter_func: Optional function to filter rows (receives DataFrame chunk)
        chunk_size: Number of rows per chunk
    
    Returns:
        dict with aggregated metrics
    """
    parquet_file = pq.ParquetFile(works_filepath)
    accumulator = MetricsAccumulator()
    
    total_rows = parquet_file.metadata.num_rows
    chunks_processed = 0
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        
        # Parse JSON fields if needed
        if 'open_access' in df_chunk.columns:
            df_chunk['oa_status'] = df_chunk['open_access'].apply(
                lambda x: safe_get(parse_json_field(x), 'oa_status', default='closed')
            )
        
        # Extract publication year if needed
        if 'publication_year' not in df_chunk.columns and 'biblio' in df_chunk.columns:
            df_chunk['publication_year'] = df_chunk['biblio'].apply(
                lambda x: safe_get(parse_json_field(x), 'year')
            )
        
        # Apply filter if provided
        if filter_func is not None:
            df_chunk = filter_func(df_chunk)
        
        if len(df_chunk) > 0:
            accumulator.add_batch(df_chunk)
        
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            print(f"    Processed {chunks_processed * chunk_size:,} / {total_rows:,} rows...")
    
    return accumulator.get_metrics()

def get_year_range(works_filepath):
    """
    Detect the range of years available in the works data.
    Returns (min_year, max_year)
    """
    parquet_file = pq.ParquetFile(works_filepath)
    years = set()
    
    # Sample first few batches to get year range
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=100000)):
        df_chunk = batch.to_pandas()
        
        # Extract publication year
        if 'publication_year' not in df_chunk.columns and 'biblio' in df_chunk.columns:
            df_chunk['publication_year'] = df_chunk['biblio'].apply(
                lambda x: safe_get(parse_json_field(x), 'year')
            )
        
        if 'publication_year' in df_chunk.columns:
            chunk_years = pd.to_numeric(df_chunk['publication_year'], errors='coerce').dropna()
            years.update(chunk_years.unique())
        
        # Sample first 500k rows to get good coverage
        if i >= 4:
            break
    
    if years:
        return int(min(years)), int(max(years))
    else:
        return 2000, 2025  # Default fallback

def calculate_annual_metrics_chunked(works_filepath, start_year=None, end_year=None):
    """
    Calculate metrics for each year and for the entire period.
    Processes data in chunks to avoid loading all to memory.
    If start_year/end_year not provided, auto-detects from data.
    
    Returns:
        - annual_metrics: DataFrame with metrics per year
        - period_metrics: dict with metrics for entire period
    """
    # Auto-detect year range if not provided
    if start_year is None or end_year is None:
        print("  → Detecting year range...")
        detected_start, detected_end = get_year_range(works_filepath)
        start_year = start_year or detected_start
        end_year = end_year or detected_end
        print(f"  → Year range: {start_year}-{end_year}")
    
    print("  → Calculating annual metrics...")
    
    # Calculate metrics for each year
    annual_data = []
    for year in range(start_year, end_year + 1):
        print(f"    Year {year}...")
        filter_func = lambda df, y=year: df[df['publication_year'] == y]
        metrics = process_works_in_chunks(works_filepath, filter_func)
        metrics['year'] = year
        annual_data.append(metrics)
    
    annual_metrics_df = pd.DataFrame(annual_data)
    
    # Calculate metrics for entire period
    print(f"    Period {start_year}-{end_year}...")
    filter_func = lambda df: df[
        (df['publication_year'] >= start_year) & 
        (df['publication_year'] <= end_year)
    ]
    period_metrics = process_works_in_chunks(works_filepath, filter_func)
    period_metrics['period'] = f'{start_year}-{end_year}'
    
    return annual_metrics_df, period_metrics

def calculate_journal_metrics_chunked(works_filepath, journals_df, journal_id, start_year=None, end_year=None):
    """
    Calculate metrics for a specific journal using chunk processing.
    """
    # Get journal metadata
    journal_info = journals_df[journals_df['id'] == journal_id]
    
    if len(journal_info) == 0:
        return None, None
    
    journal_info = journal_info.iloc[0]
    
    # Extract indexing information
    is_scopus = safe_get(journal_info, 'is_indexed_in_scopus', default=False)
    is_core = safe_get(journal_info, 'is_core', default=False)
    is_doaj = safe_get(journal_info, 'is_in_doaj', default=False)
    
    journal_indexing = {
        'is_scopus': bool(is_scopus),
        'is_core': bool(is_core),
        'is_doaj': bool(is_doaj)
    }
    
    # Get year range if not provided
    if start_year is None or end_year is None:
        start_year, end_year = get_year_range(works_filepath)
    
    # Annual metrics
    annual_data = []
    for year in range(start_year, end_year + 1):
        year_filter = lambda df, jid=journal_id, y=year: df[(df['journal_id'] == jid) & (df['publication_year'] == y)]
        metrics = process_works_in_chunks(works_filepath, year_filter)
        metrics['year'] = year
        metrics['journal_id'] = journal_id
        # Add indexing info to annual metrics
        metrics.update(journal_indexing)
        annual_data.append(metrics)
    
    annual_metrics_df = pd.DataFrame(annual_data)
    
    # Period metrics
    period_filter = lambda df, jid=journal_id: df[
        (df['journal_id'] == jid) & 
        (df['publication_year'] >= start_year) & 
        (df['publication_year'] <= end_year)
    ]
    period_metrics = process_works_in_chunks(works_filepath, period_filter)
    period_metrics['journal_id'] = journal_id
    period_metrics['period'] = f'{start_year}-{end_year}'
    # Add indexing info to period metrics
    period_metrics.update(journal_indexing)
    
    return annual_metrics_df, period_metrics

def calculate_country_metrics_chunked(works_filepath, journals_df, country_code, start_year=None, end_year=None):
    """
    Calculate metrics for a specific country using chunk processing.
    """
    country_journals = journals_df[journals_df['country_code'] == country_code]
    
    if len(country_journals) == 0:
        return None, None, None
    
    # Get year range if not provided
    if start_year is None or end_year is None:
        start_year, end_year = get_year_range(works_filepath)
    
    num_journals = len(country_journals)
    journal_ids = country_journals['id'].tolist()
    
    # Journal indexing metrics
    pct_scopus = (country_journals.apply(lambda x: safe_get(x, 'is_indexed_in_scopus', default=False), axis=1).sum() / num_journals) * 100
    pct_core = (country_journals.apply(lambda x: safe_get(x, 'is_core', default=False), axis=1).sum() / num_journals) * 100
    pct_doaj = (country_journals.apply(lambda x: safe_get(x, 'is_in_doaj', default=False), axis=1).sum() / num_journals) * 100
    
    journal_metrics = {
        'num_journals': num_journals,
        'pct_scopus': round(pct_scopus, 2),
        'pct_core': round(pct_core, 2),
        'pct_doaj': round(pct_doaj, 2)
    }
    
    print(f"    Processing country {country_code} ({num_journals} journals)...")
    
    # Annual metrics
    annual_data = []
    for year in range(start_year, end_year + 1):
        year_filter = lambda df, jids=journal_ids, y=year: df[
            (df['journal_id'].isin(jids)) & 
            (df['publication_year'] == y)
        ]
        metrics = process_works_in_chunks(works_filepath, year_filter)
        metrics['year'] = year
        metrics['country_code'] = country_code
        annual_data.append(metrics)
    
    annual_metrics_df = pd.DataFrame(annual_data)
    
    # Period metrics
    period_filter = lambda df, jids=journal_ids: df[
        (df['journal_id'].isin(jids)) & 
        (df['publication_year'] >= start_year) & 
        (df['publication_year'] <= end_year)
    ]
    period_metrics = process_works_in_chunks(works_filepath, period_filter)
    period_metrics.update(journal_metrics)
    period_metrics['country_code'] = country_code
    period_metrics['period'] = '2021-2025'
    
    return annual_metrics_df, period_metrics, journal_metrics

def calculate_latam_metrics_chunked(works_filepath, journals_df, start_year=None, end_year=None):
    """
    Calculate metrics for all LATAM using chunk processing.
    """
    num_journals = len(journals_df)
    
    if num_journals == 0:
        return None, None, None
    
    # Get year range if not provided
    if start_year is None or end_year is None:
        start_year, end_year = get_year_range(works_filepath)
    
    # Journal indexing metrics
    pct_scopus = (journals_df.apply(lambda x: safe_get(x, 'is_indexed_in_scopus', default=False), axis=1).sum() / num_journals) * 100
    pct_core = (journals_df.apply(lambda x: safe_get(x, 'is_core', default=False), axis=1).sum() / num_journals) * 100
    pct_doaj = (journals_df.apply(lambda x: safe_get(x, 'is_in_doaj', default=False), axis=1).sum() / num_journals) * 100
    
    journal_metrics = {
        'num_journals': num_journals,
        'pct_scopus': round(pct_scopus, 2),
        'pct_core': round(pct_core, 2),
        'pct_doaj': round(pct_doaj, 2)
    }
    
    print(f"    Processing LATAM ({num_journals} journals)...")
    
    # Annual metrics
    annual_data = []
    for year in range(start_year, end_year + 1):
        year_filter = lambda df, y=year: df[df['publication_year'] == y]
        metrics = process_works_in_chunks(works_filepath, year_filter)
        metrics['year'] = year
        annual_data.append(metrics)
    
    annual_metrics_df = pd.DataFrame(annual_data)
    
    # Period metrics
    period_filter = lambda df: df[
        (df['publication_year'] >= start_year) & 
        (df['publication_year'] <= end_year)
    ]
    period_metrics = process_works_in_chunks(works_filepath, period_filter)
    period_metrics.update(journal_metrics)
    period_metrics['period'] = f'{start_year}-{end_year}'
    
    return annual_metrics_df, period_metrics, journal_metrics

def compute_and_cache_all_metrics(works_filepath, journals_filepath, force_recalculate=False):
    """
    Master function to compute and cache all performance metrics.
    Uses chunk-based processing to avoid loading all data to memory.
    
    Generates:
    - Annual metrics tables (2021-2025) for journals, countries, and LATAM
    - Period metrics (2021-2025 aggregate) for journals, countries, and LATAM
    
    Args:
        works_filepath: Path to works parquet file
        journals_filepath: Path to journals parquet file
        force_recalculate: If True, ignore cache and recalculate everything
    
    Returns:
        dict with all computed metrics
    """
    cache_dir = get_cache_dir()
    
    print("⚙️  Loading journals data...")
    try:
        journals_df = pd.read_parquet(journals_filepath)
    except Exception as e:
        print(f"⚠️  Error loading journals data: {e}")
        return None
    
    if journals_df.empty:
        print("⚠️  No journals data available")
        return None
    
    print(f"✓ Loaded {len(journals_df):,} journals")
    
    # Verify works file exists
    if not os.path.exists(works_filepath):
        print(f"⚠️  Works file not found: {works_filepath}")
        return None
    
    parquet_file = pq.ParquetFile(works_filepath)
    total_works = parquet_file.metadata.num_rows
    print(f"✓ Works file contains {total_works:,} articles")
    
    # Detect year range
    print("\n⚙️  Detecting year range in data...")
    start_year, end_year = get_year_range(works_filepath)
    print(f"✓ Year range: {start_year}-{end_year}")
    
    # Calculate metrics at all levels
    print("\n⚙️  Computing metrics (chunk-based processing)...")
    
    # 1. LATAM level (process first as it's the broadest)
    print("\n📊 LATAM metrics...")
    latam_annual, latam_period, _ = calculate_latam_metrics_chunked(works_filepath, journals_df, start_year, end_year)
    
    if latam_annual is not None:
        latam_annual.to_parquet(cache_dir / 'metrics_latam_annual.parquet', index=False)
        print(f"  ✓ Saved LATAM annual metrics: {len(latam_annual)} years")
    
    if latam_period is not None:
        latam_period_df = pd.DataFrame([latam_period])
        latam_period_df.to_parquet(cache_dir / 'metrics_latam_period.parquet', index=False)
        print(f"  ✓ Saved LATAM period metrics")
    
    # 2. Country level
    print("\n📊 Country metrics...")
    country_annual_list = []
    country_period_list = []
    
    for country_code in journals_df['country_code'].unique():
        annual, period, _ = calculate_country_metrics_chunked(works_filepath, journals_df, country_code, start_year, end_year)
        
        if annual is not None:
            country_annual_list.append(annual)
        
        if period is not None:
            country_period_list.append(period)
    
    country_annual_df = pd.concat(country_annual_list, ignore_index=True) if country_annual_list else pd.DataFrame()
    country_period_df = pd.DataFrame(country_period_list) if country_period_list else pd.DataFrame()
    
    if not country_annual_df.empty:
        country_annual_df.to_parquet(cache_dir / 'metrics_country_annual.parquet', index=False)
        print(f"  ✓ Saved country annual metrics: {len(country_annual_df)} rows")
    
    if not country_period_df.empty:
        country_period_df.to_parquet(cache_dir / 'metrics_country_period.parquet', index=False)
        print(f"  ✓ Saved country period metrics: {len(country_period_df)} countries")
    
    # 3. Journal level (most granular - can be slow)
    print("\n📊 Journal metrics...")
    print(f"  Processing {len(journals_df)} journals...")
    
    journal_annual_list = []
    journal_period_list = []
    
    for idx, journal_id in enumerate(journals_df['id'].unique(), 1):
        if idx % 50 == 0:
            print(f"    Progress: {idx}/{len(journals_df)} journals...")
        
        annual, period = calculate_journal_metrics_chunked(works_filepath, journals_df, journal_id, start_year, end_year)
        
        if annual is not None and len(annual) > 0:
            journal_annual_list.append(annual)
        
        if period is not None:
            journal_period_list.append(period)
    
    journal_annual_df = pd.concat(journal_annual_list, ignore_index=True) if journal_annual_list else pd.DataFrame()
    journal_period_df = pd.DataFrame(journal_period_list) if journal_period_list else pd.DataFrame()
    
    if not journal_annual_df.empty:
        journal_annual_df.to_parquet(cache_dir / 'metrics_journal_annual.parquet', index=False)
        print(f"  ✓ Saved journal annual metrics: {len(journal_annual_df)} rows")
    
    if not journal_period_df.empty:
        journal_period_df.to_parquet(cache_dir / 'metrics_journal_period.parquet', index=False)
        print(f"  ✓ Saved journal period metrics: {len(journal_period_df)} journals")
    
    print("\n✅ All metrics computed and cached successfully!")
    
    return {
        'journal_annual': journal_annual_df,
        'journal_period': journal_period_df,
        'country_annual': country_annual_df,
        'country_period': country_period_df,
        'latam_annual': latam_annual,
        'latam_period': latam_period
    }

def load_cached_metrics(level, metric_type='period'):
    """
    Load cached metrics.
    
    Args:
        level: 'journal', 'country', or 'latam'
        metric_type: 'annual' or 'period'
    
    Returns:
        DataFrame with metrics or None if not found
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f'metrics_{level}_{metric_type}.parquet'
    
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None
