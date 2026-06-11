import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect

# Cargar variables de entorno
load_dotenv()

BASE_PATH = Path(__file__).parent.parent
# Forzar carga de .env desde el root
env_path = BASE_PATH / '.env'
load_dotenv(dotenv_path=env_path)

DATA_DIR = BASE_PATH / 'data'
CACHE_SCOPUS_DIR = DATA_DIR / 'cache_scopus'
RAW_SCOPUS_DIR = CACHE_SCOPUS_DIR / 'raw'
PROCESSED_SCOPUS_DIR = CACHE_SCOPUS_DIR / 'processed'

PROCESSED_SCOPUS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de conexión a ClickHouse
CH_HOST = os.environ.get('CH_HOST', 'localhost')
CH_PORT = int(os.environ.get('CH_PORT', 8124))
CH_USER = os.environ.get('CH_USER', 'default')
CH_PASSWORD = os.environ.get('CH_PASSWORD', '')
CH_DATABASE = os.environ.get('CH_DATABASE', 'rag')

def get_ch_client():
    """Retorna un cliente conectado a ClickHouse."""
    is_secure = (CH_PORT == 8124)
    print(f"[*] Intentando conectar a ClickHouse en {CH_HOST}:{CH_PORT} (secure={is_secure})...", flush=True)
    try:
        client = clickhouse_connect.get_client(
            host=CH_HOST,
            port=CH_PORT,
            username=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE,
            secure=is_secure,
            verify=False,
            connect_timeout=10,
            send_receive_timeout=60
        )
        print("[OK] Conectado a ClickHouse (seguro).", flush=True)
        return client
    except Exception as e:
        print(f"[!] Error conectando seguro: {e}", flush=True)
        if is_secure:
            print("[*] Reintentando conexión no segura...", flush=True)
            try:
                client = clickhouse_connect.get_client(
                    host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD,
                    database=CH_DATABASE, secure=False, verify=False,
                    connect_timeout=10
                )
                print("[OK] Conectado a ClickHouse (no seguro).", flush=True)
                return client
            except Exception as e2:
                print(f"[!] Error conectando no seguro: {e2}", flush=True)
        raise e

def procesar_scopus(input_parquet_path):
    """
    Lee un archivo de resultados de Scopus, extrae los DOIs, y cruza 
    contra works_flat en ClickHouse para traernos el registro completo y el abstract.
    """
    input_path = Path(input_parquet_path)
    if not input_path.exists():
        print(f"[!] Archivo no encontrado: {input_path}", flush=True)
        return None
        
    print(f"[*] Cargando archivo Scopus: {input_path.name}", flush=True)
    df_scopus = pd.read_parquet(input_path)
    print(f"   -> {len(df_scopus)} registros cargados.", flush=True)
    
    # Extraer y limpiar DOIs
    if 'doi' not in df_scopus.columns:
        print("[!] El dataset no contiene columna 'doi'. Imposible cruzar.", flush=True)
        return None
        
    # Filtrar nulos y limpiar
    scopus_dois = df_scopus['doi'].dropna().astype(str).str.strip().str.lower()
    
    # OpenAlex DOIs en works_flat generalmente tienen formato https://doi.org/10...
    # Vamos a crear la lista de URLs de DOI
    openalex_dois = "https://doi.org/" + scopus_dois
    doi_list = openalex_dois.tolist()
    
    print(f"[*] {len(doi_list)} DOIs válidos extraídos para cruce.", flush=True)
    
    if len(doi_list) == 0:
        print("[!] No hay DOIs para buscar.", flush=True)
        return None
        
    client = get_ch_client()
    
    print("[*] Consultando base de datos works_flat en ClickHouse...", flush=True)
    # Como la lista de DOIs puede ser larga (miles), usamos la funcionalidad de parametros
    # O para listas largas podemos dividirlas en chunks si es necesario, pero < 10k deberia aguantar.
    
    # Para traer todo el registro, usamos SELECT * pero omitiendo embeddings si pesan mucho, 
    # o seleccionamos las columnas que necesitamos. Si pedimos todo, SELECT * FINAL.
    
    # Evitamos usar FINAL porque colapsa la memoria o tarda horas. 
    # Mejor traemos todo sin FINAL y limpiamos duplicados en pandas si los hay.
    try:
        print(f"[*] Ejecutando query_df con {len(doi_list)} parametros en chunks de 5000...", flush=True)
        chunk_size = 5000
        all_dfs = []
        for i in range(0, len(doi_list), chunk_size):
            chunk = doi_list[i:i + chunk_size]
            # Formateamos la lista de strings para inyectarla directo en SQL y forzar POST body en el driver
            formatted_dois = ", ".join(f"'{str(doi).replace(chr(39), chr(39)+chr(39))}'" for doi in chunk)
            
            chunk_query = f"""
                SELECT *
                FROM works_flat
                WHERE doi IN ({formatted_dois})
            """
            
            df_chunk = client.query_df(chunk_query)
            if len(df_chunk) > 0:
                all_dfs.append(df_chunk)
                
        if all_dfs:
            df_openalex = pd.concat(all_dfs, ignore_index=True)
        else:
            df_openalex = pd.DataFrame()
        
        # ClickHouse puede devolver duplicados si hay un ReplacingMergeTree sin FINAL
        if 'id' in df_openalex.columns:
            df_openalex = df_openalex.drop_duplicates(subset=['id'], keep='last')
            
        print(f"   -> {len(df_openalex)} registros únicos encontrados en ClickHouse.", flush=True)
        
        # --- COBERTURA ---
        # Artículos en Scopus sin DOI
        sin_doi = int(df_scopus['doi'].isna().sum())
        # Artículos con DOI que NO hicieron match en OpenAlex
        dois_en_openalex = set(df_openalex['doi'].dropna().str.lower().str.replace('https://doi.org/', '', regex=False)) if len(df_openalex) > 0 else set()
        df_scopus_con_doi = df_scopus.dropna(subset=['doi']).copy()
        df_scopus_con_doi['doi_clean'] = df_scopus_con_doi['doi'].astype(str).str.strip().str.lower()
        df_no_match = df_scopus_con_doi[~df_scopus_con_doi['doi_clean'].isin(dois_en_openalex)]
        
        coverage = {
            "total_scopus_raw": int(len(df_scopus)),
            "sin_doi": sin_doi,
            "con_doi": int(len(df_scopus) - sin_doi),
            "matched_openalex": int(len(df_openalex)),
            "no_match_openalex": int(len(df_no_match)),
            "cobertura_pct": round(len(df_openalex) / max(len(df_scopus) - sin_doi, 1) * 100, 1)
        }
        
        # Guardar cobertura
        coverage_path = PROCESSED_SCOPUS_DIR / (input_path.stem + "_coverage.json")
        import json
        with open(coverage_path, 'w', encoding='utf-8') as f:
            json.dump(coverage, f, indent=4, ensure_ascii=False)
        print(f"[*] Cobertura: {coverage['matched_openalex']}/{coverage['con_doi']} DOIs encontrados en OpenAlex ({coverage['cobertura_pct']}%)", flush=True)
        
        # Guardar artículos no encontrados en OpenAlex
        if len(df_no_match) > 0:
            unmatched_path = PROCESSED_SCOPUS_DIR / (input_path.stem + "_no_en_openalex.parquet")
            df_no_match.drop(columns=['doi_clean'], errors='ignore').to_parquet(unmatched_path, index=False)
            print(f"[*] {len(df_no_match)} artículos sin match guardados en: {unmatched_path.name}", flush=True)
        
        if len(df_openalex) > 0:
            # Guardamos los registros completos de OpenAlex
            output_filename = input_path.stem + "_openalex.parquet"
            output_path = PROCESSED_SCOPUS_DIR / output_filename
            
            df_openalex.to_parquet(output_path, index=False)
            print(f"[OK] Registros combinados/OpenAlex guardados en: {output_path}", flush=True)
            return output_path
        else:
            print("[!] Ningun DOI hizo match en OpenAlex.", flush=True)
            return None
            
    except Exception as e:
        print(f"[!] Error al consultar ClickHouse: {e}", flush=True)
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Procesa datos de Scopus y cruza con OpenAlex/ClickHouse.")
    parser.add_argument("--input", required=True, help="Ruta al archivo .parquet generado por scopus_downloader.py")
    
    args = parser.parse_args()
    procesar_scopus(args.input)
