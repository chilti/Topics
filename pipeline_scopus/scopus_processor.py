import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import clickhouse_connect
import re
import numpy as np

def wildcard_to_regex(term):
    term = re.escape(term)
    term = term.replace(r'\*', r'.*').replace(r'\?', r'.')
    pattern = r'\b' + term
    if not pattern.endswith('.*'):
        pattern += r'\b'
    return pattern

def check_proximity(text, term1, term2, distance):
    if not isinstance(text, str):
        return False
    words = re.findall(r'\b\w+\b', text.lower())
    p1 = re.compile(wildcard_to_regex(term1).lower())
    p2 = re.compile(wildcard_to_regex(term2).lower())
    
    indices1 = [idx for idx, w in enumerate(words) if p1.match(w)]
    indices2 = [idx for idx, w in enumerate(words) if p2.match(w)]
    
    for i1 in indices1:
        for i2 in indices2:
            if i1 == i2:
                continue
            if abs(i1 - i2) <= distance:
                return True
    return False

def match_proximity_pattern(text, pattern_str):
    m = re.match(r'\((.*?)\s+W/(\d+)\s+(.*?)\)', pattern_str, re.IGNORECASE)
    if m:
        t1, dist, t2 = m.groups()
        return check_proximity(text, t1, t2, int(dist))
    return False

def compile_terms(terms_list):
    regex_terms = []
    proximity_terms = []
    
    for t in terms_list:
        t = t.strip()
        if not t:
            continue
        if ' W/' in t.upper():
            proximity_terms.append(t)
        else:
            regex_terms.append(wildcard_to_regex(t))
            
    pattern = None
    if regex_terms:
        pattern = re.compile('|'.join(regex_terms), re.IGNORECASE)
        
    return pattern, proximity_terms

def filtrar_exclusiones_locales(df, exclusions):
    """
    Aplica una lista de reglas de exclusión sobre el DataFrame de OpenAlex.
    df: DataFrame con columnas 'title', 'abstract', 'concepts', 'keywords', etc.
    exclusions: List of dicts representing exclusion rules.
    """
    if df.empty or not exclusions:
        return df
        
    # Creamos representaciones de texto limpio para evaluar
    titles = df['title'].fillna('').astype(str)
    abstracts = df['abstract'].fillna('').astype(str)
    
    def list_to_str(val):
        if isinstance(val, (list, np.ndarray)):
            return " ".join(str(x) for x in val)
        return str(val) if pd.notna(val) else ""
        
    keywords_col = df.get('keywords', pd.Series(dtype=str)).apply(list_to_str)
    concepts_col = df.get('concepts', pd.Series(dtype=str)).apply(list_to_str)
    
    # Combinado completo para búsquedas de tipo TITLE-ABS-KEY
    title_abs_key = (titles + " " + abstracts + " " + keywords_col + " " + concepts_col).str.lower()
    title_abs = (titles + " " + abstracts).str.lower()
    
    # Empezamos con una máscara donde todos los registros son válidos (True)
    keep_mask = pd.Series(True, index=df.index)
    
    for idx, rule in enumerate(exclusions):
        excl_terms = rule.get('exclude_terms', [])
        reincl_terms = rule.get('reinclude_terms', [])
        exclude_vete = rule.get('exclude_vete', False)
        
        print(f"[*] Aplicando regla de exclusión {idx+1} localmente:")
        print(f"    - Excluir términos: {len(excl_terms)}")
        print(f"    - Re-incluir términos: {len(reincl_terms)}")
        
        # 1. Evaluar exclusiones
        excl_pattern, excl_prox = compile_terms(excl_terms)
        
        # Si la regla tiene exclude_vete, es el bloque de animales/veterinaria
        is_animal_block = exclude_vete or any(x in str(excl_terms).lower() for x in ['murine', 'rodent', 'mouse'])
        target_text = title_abs if is_animal_block else title_abs_key
        
        if excl_pattern or excl_prox:
            # Mask de filas que contienen exclusión
            exclude_match = pd.Series(False, index=df.index)
            if excl_pattern:
                exclude_match = exclude_match | target_text.str.contains(excl_pattern.pattern, case=False, na=False)
            if excl_prox:
                exclude_match = exclude_match | target_text.apply(lambda x: any(match_proximity_pattern(x, pt) for pt in excl_prox))
                
            # 2. Evaluar re-inclusiones
            if reincl_terms and exclude_match.any():
                reincl_pattern, reincl_prox = compile_terms(reincl_terms)
                reinclude_match = pd.Series(False, index=df.index)
                
                target_reincl = title_abs_key if not is_animal_block else title_abs
                
                # Evaluar solo sobre las filas pre-excluidas para ahorrar recursos
                sub_target = target_reincl[exclude_match]
                sub_reinclude = pd.Series(False, index=sub_target.index)
                if reincl_pattern:
                    sub_reinclude = sub_reinclude | sub_target.str.contains(reincl_pattern.pattern, case=False, na=False)
                if reincl_prox:
                    sub_reinclude = sub_reinclude | sub_target.apply(lambda x: any(match_proximity_pattern(x, pt) for pt in reincl_prox))
                    
                reinclude_match.loc[sub_reinclude.index] = sub_reinclude
                
                # Excluir solo si coincide con exclusión Y NO coincide con re-inclusión
                final_excl_mask = exclude_match & ~reinclude_match
            else:
                final_excl_mask = exclude_match
                
            keep_mask = keep_mask & ~final_excl_mask
            
    filtered_df = df[keep_mask]
    excluded_count = len(df) - len(filtered_df)
    print(f"[OK] Filtrado local finalizado. Excluidos {excluded_count} de {len(df)} registros. Conservados {len(filtered_df)}.")
    return filtered_df


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
        
        matched_scopus = int(len(df_scopus_con_doi) - len(df_no_match))
        
        coverage = {
            "total_scopus_raw": int(len(df_scopus)),
            "sin_doi": sin_doi,
            "con_doi": int(len(df_scopus) - sin_doi),
            "matched_openalex": matched_scopus,
            "no_match_openalex": int(len(df_no_match)),
            "cobertura_pct": round(matched_scopus / max(len(df_scopus) - sin_doi, 1) * 100, 1)
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
            
        # Cargar exclusiones si existen en el JSON de metadata
        exclusions = []
        meta_path = input_path.with_suffix('.json')
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    exclusions = meta_data.get('exclusions', [])
                    if exclusions:
                        print(f"[*] Encontradas {len(exclusions)} reglas de exclusión en el archivo de metadatos.", flush=True)
            except Exception as e:
                print(f"[!] Error al leer metadatos de exclusión: {e}", flush=True)
        
        if len(df_openalex) > 0:
            if exclusions:
                print("[*] Aplicando exclusiones locales sobre el corpus de OpenAlex...", flush=True)
                df_openalex = filtrar_exclusiones_locales(df_openalex, exclusions)
                
            if len(df_openalex) > 0:
                # Guardamos los registros completos de OpenAlex
                output_filename = input_path.stem + "_openalex.parquet"
                output_path = PROCESSED_SCOPUS_DIR / output_filename
                
                df_openalex.to_parquet(output_path, index=False)
                print(f"[OK] Registros filtrados y cruzados con OpenAlex guardados en: {output_path}", flush=True)
                return output_path
            else:
                print("[!] Todos los registros fueron excluidos por los filtros locales.", flush=True)
                return None
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
