import pyalex
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import datetime
import time
import json

# Configure PyAlex
pyalex.config.email = "jlja@ciencias.unam.mx" # User email from the notebook

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PARQUET_FILE = os.path.join(DATA_DIR, 'latin_american_journals.parquet')
WORKS_PARQUET_FILE = os.path.join(DATA_DIR, 'latin_american_works.parquet')

# List of Latin American country codes (ISO 2-letter codes)
LATAM_COUNTRIES = [
    'MX', # Mexico
    'BR', # Brazil
    'AR', # Argentina
    'CL', # Chile
    'CO', # Colombia
    'PE', # Peru
    'VE', # Venezuela
    'EC', # Ecuador
    'CU', # Cuba
    'UY', # Uruguay
    'CR', # Costa Rica
    'BO', # Bolivia
    'DO', # Dominican Republic
    'GT', # Guatemala
    'PY', # Paraguay
    'SV', # El Salvador
    'HN', # Honduras
    'NI', # Nicaragua
    'PA', # Panama
    'PR', # Puerto Rico (often included in LATAM analysis)
]

def fetch_journals_by_country(country_code):
    """
    Fetches sources (journals) from OpenAlex for a specific country code.
    """
    try:
        # Filter sources by country
        # We are looking for journals (type='journal') hosted in the specified country
        query = Sources().filter(country_code=country_code, type='journal')
        # Use paginate to get all results (generator yields pages)
        results = query.paginate(per_page=200)
        
        journals_data = []
        for page in results:
            for source in page:
                # Store complete OpenAlex record
                journal = dict(source)
                # Add metadata
                journal['download_date'] = datetime.datetime.now().isoformat()
                journals_data.append(journal)
            
        return journals_data
    except Exception as e:
        print(f"Error fetching data for {country_code}: {e}")
        return []

def fetch_works_for_journal(journal_id, journal_name):
    """
    Fetches all works (articles) for a specific journal.
    Implements rate limiting to comply with OpenAlex policies (max 10 requests/second for polite pool).
    """
    try:
        print(f"  Fetching works for: {journal_name}...")
        
        # Query works by journal (primary_location.source.id)
        query = Works().filter(primary_location={'source': {'id': journal_id}})
        
        # Use paginate to get all results
        results = query.paginate(per_page=200)
        
        works_data = []
        page_count = 0
        
        for page in results:
            page_count += 1
            
            # Rate limiting: OpenAlex polite pool allows ~10 req/sec
            # We'll be conservative with 0.15s between pages (~6.6 req/sec)
            if page_count > 1:
                time.sleep(0.15)
            
            for work in page:
                # Store complete OpenAlex record
                work_record = dict(work)
                # Add metadata for easier filtering
                work_record['journal_id'] = journal_id
                work_record['journal_name'] = journal_name
                work_record['download_date'] = datetime.datetime.now().isoformat()
                works_data.append(work_record)
            
            # Progress indicator every 5 pages
            if page_count % 5 == 0:
                print(f"    Processed {page_count} pages, {len(works_data)} works so far...")
        
        print(f"  Found {len(works_data)} works for {journal_name}.")
        return works_data
        
    except Exception as e:
        print(f"  Error fetching works for {journal_name}: {e}")
        return []

def get_downloaded_journal_ids():
    """
    Returns a set of journal IDs that already have works downloaded.
    This allows resuming downloads without re-downloading existing data.
    """
    if not os.path.exists(WORKS_PARQUET_FILE):
        return set()
    
    try:
        # Read just the journal_id column to check what's already downloaded
        df = pd.read_parquet(WORKS_PARQUET_FILE, columns=['journal_id'])
        downloaded_ids = set(df['journal_id'].unique())
        print(f"Found {len(downloaded_ids)} journals already downloaded.")
        return downloaded_ids
    except Exception as e:
        print(f"Warning: Could not read existing works file: {e}")
        return set()

def update_data(include_works=True, resume=True, update_journals=False):
    """
    Main function to update the dataset for all LATAM countries.
    
    Args:
        include_works: If True, also downloads all works (articles) for each journal.
        resume: If True, skips journals that already have works downloaded.
        update_journals: If True, re-downloads journal metadata. Default False to avoid unnecessary API calls.
    """
    all_journals = []
    print("Starting data update from OpenAlex...")
    
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Check if we should update journals
    if update_journals or not os.path.exists(PARQUET_FILE):
        print("\n" + "="*60)
        print("DOWNLOADING JOURNAL METADATA")
        print("="*60)
        
        for country in LATAM_COUNTRIES:
            print(f"Fetching journals for {country}...")
            country_journals = fetch_journals_by_country(country)
            all_journals.extend(country_journals)
            print(f"Found {len(country_journals)} journals for {country}.")

        if all_journals:
            df = pd.DataFrame(all_journals)
            
            # Save journals to parquet
            print(f"Saving {len(df)} journal records to {PARQUET_FILE}...")
            df.to_parquet(PARQUET_FILE, index=False)
            print("Journal data update complete.")
        else:
            print("No journal data found.")
            return 0
    else:
        print("\n" + "="*60)
        print("LOADING EXISTING JOURNAL METADATA")
        print("="*60)
        print(f"Using existing journal file: {PARQUET_FILE}")
        df = pd.read_parquet(PARQUET_FILE)
        all_journals = df.to_dict('records')
        print(f"Loaded {len(all_journals)} journals from cache.")

    if all_journals:
        
        # Download works if requested
        if include_works:
            print("\n" + "="*60)
            print("Starting Works (articles) download...")
            print(f"This will download articles for {len(df)} journals.")
            
            # Check for resume capability
            downloaded_journal_ids = set()
            if resume:
                downloaded_journal_ids = get_downloaded_journal_ids()
                if downloaded_journal_ids:
                    print(f"RESUME MODE: Skipping {len(downloaded_journal_ids)} already downloaded journals.")
            
            journals_to_process = [j for j in all_journals if j['id'] not in downloaded_journal_ids]
            
            if not journals_to_process:
                print("All journals already downloaded! Nothing to do.")
                return len(df)
            
            print(f"Will process {len(journals_to_process)} journals.")
            print("This may take a considerable amount of time.")
            print("="*60 + "\n")
            
            # Incremental writing with ParquetWriter to avoid reading file into memory
            batch_works = []
            batch_size = 1  # Write every journal
            writer = None
            
            # If resuming, we need to append to existing file
            append_mode = resume and len(downloaded_journal_ids) > 0
            
            for idx, journal in enumerate(journals_to_process, 1):
                print(f"[{idx}/{len(journals_to_process)}] Processing {journal['display_name']}...")
                journal_works = fetch_works_for_journal(journal['id'], journal['display_name'])
                batch_works.extend(journal_works)
                
                # Write batch to parquet and clear memory every batch_size journals
                if idx % batch_size == 0 or idx == len(journals_to_process):
                    if batch_works:
                        batch_df = pd.DataFrame(batch_works)
                        
                        # PyArrow strict schema enforcement with nested structs is very fragile for evolving data like OpenAlex
                        # Best practice for robustness: Convert all complex types (list, dict) to JSON strings
                        # This ensures the schema is always simple (mostly strings/numbers) and never fails on new nested fields
                        for col in batch_df.columns:
                            # Check if column contains complex types
                            if batch_df[col].dtype == 'object':
                                # Check a sample of non-null values to verify if it's a dict/list
                                sample = batch_df[col].dropna().head(1)
                                if not sample.empty and isinstance(sample.iloc[0], (dict, list)):
                                     # Convert entire column to json string
                                     batch_df[col] = batch_df[col].apply(lambda x: json.dumps(x) if x is not None else None)
                        
                        # Convert to PyArrow Table
                        table = pa.Table.from_pandas(batch_df)
                        
                        if writer is None:
                            # Check if we're appending to an existing file
                            if append_mode and os.path.exists(WORKS_PARQUET_FILE):
                                # Read existing schema from file
                                existing_file = pq.ParquetFile(WORKS_PARQUET_FILE)
                                robust_schema = existing_file.schema_arrow
                                print(f"  → Appending to existing file with {existing_file.metadata.num_rows:,} rows")
                                
                                # Cast new table to match existing schema
                                try:
                                    # Ensure batch has all columns from existing schema
                                    batch_df = batch_df.reindex(columns=robust_schema.names)
                                    table = pa.Table.from_pandas(batch_df, schema=robust_schema)
                                except Exception as e:
                                    print(f"  Warning: Schema mismatch ({e}), will try to adapt...")
                                    # Convert object columns to strings to match
                                    for col in batch_df.columns:
                                        if batch_df[col].dtype == 'object':
                                            batch_df[col] = batch_df[col].apply(lambda x: str(x) if pd.notnull(x) else None)
                                    batch_df = batch_df.reindex(columns=robust_schema.names)
                                    table = pa.Table.from_pandas(batch_df, schema=robust_schema)
                                
                                # Open in append mode
                                writer = pq.ParquetWriter(WORKS_PARQUET_FILE, robust_schema)
                            else:
                                # Initialize writer with the schema of the first batch
                                # Fix: Coerce Null-inferred columns to String to accommodate future data
                                new_fields = []
                                for field in table.schema:
                                    if field.type == pa.null():
                                        new_fields.append(pa.field(field.name, pa.string()))
                                    else:
                                        new_fields.append(field)
                                
                                robust_schema = pa.schema(new_fields)
                                
                                # Cast table to the robust schema
                                try:
                                    table = table.cast(robust_schema)
                                except:
                                    pass
                                
                                writer = pq.ParquetWriter(WORKS_PARQUET_FILE, robust_schema)
                                print(f"  → Created new Parquet file: {WORKS_PARQUET_FILE}")
                        
                        # Ensure table matches schema of writer (handle missing/extra columns naively)
                        if not table.schema.equals(writer.schema):
                            # Reindex columns to match writer schema (adds missing as null, drops extra)
                            batch_df = batch_df.reindex(columns=writer.schema.names)
                            # Re-apply json conversion to be safe on new nulls
                            
                            # Fallback: recreate table
                            try:
                                table = pa.Table.from_pandas(batch_df, schema=writer.schema)
                            except Exception as e:
                                # Fallback 2: If schema validation fails (e.g. type mismatch), convert confusing columns to string
                                print(f"  Warning: Schema mismatch for batch ({e}). converting object columns to string.")
                                
                                # Iterate over writer schema to find string columns and force conversion
                                for field in writer.schema:
                                    if field.name in batch_df.columns:
                                        # If target is string, ensure input is string
                                        if pa.types.is_string(field.type):
                                            batch_df[field.name] = batch_df[field.name].astype(str)
                                            # Handle "None" stringification if necessary, but astype(str) usually makes 'None' or 'nan'
                                            # Better: apply json.dumps if regex/complex? Or just str()
                                            
                                # Also, for any object column in current batch that isn't in writer schema (dropped above) 
                                # or is object but writer expects something else (unlikely if first batch defined it)
                                
                                # Brute force: Convert ANY object column in batch that causes issues to string representation
                                # But we must respect writer schema.
                                # If writer schema expects "string" for a column (because first batch had json string),
                                # but current batch has "None" (null) -> PyArrow handles null to string fine.
                                # The error "Invalid null value" for "apc_list" suggests apc_list in writer schema might be Struct?
                                # If first batch had apc_list as None, schema might be Null or something else?
                                # No, if first batch had apc_list as complex dict, I converted it to json string.
                                # So writer schema expects String. 
                                # If current batch has apc_list as None (NaN), astype(str) makes "nan".
                                
                                # Let's convert ALL object columns to string if they are not null, to satisfy "String" expectation
                                for col in batch_df.columns:
                                     if batch_df[col].dtype == 'object':
                                         batch_df[col] = batch_df[col].apply(lambda x: str(x) if pd.notnull(x) else None)
                                
                                table = pa.Table.from_pandas(batch_df, schema=writer.schema)
                        
                        writer.write_table(table)
                        print(f"  → Wrote batch ({len(batch_df)} works) to disk.")
                        
                        # Clear memory
                        batch_works = []
                        del batch_df
                        del table
                    
                    print(f"\n--- Progress: {idx}/{len(journals_to_process)} journals processed ---\n")
            
            if writer:
                writer.close()
                print("Works data update complete.")
        
        return len(df)
    else:
        print("No data found.")
        return 0

def load_data():
    """
    Loads the dataset from the parquet file.
    """
    if os.path.exists(PARQUET_FILE):
        return pd.read_parquet(PARQUET_FILE)
    else:
        print("Data file not found. Please update data first.")
        return pd.DataFrame()

if __name__ == "__main__":
    update_data()
