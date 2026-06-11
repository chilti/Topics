import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Directorio base para guardar datos
DATA_DIR = Path(__file__).parent.parent / "data" / "cache_scopus" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def init_pybliometrics():
    """Asegurar que pybliometrics esté configurado y con la API key correcta."""
    # Obtenemos API key desde entorno, o usamos la nueva institucional de la UNAM
    api_key = os.environ.get('PYBLIOMETRICS_API_KEY', '72952ef45b534fa6e6f386ad415f89d9')
    os.environ['PYBLIOMETRICS_API_KEY'] = api_key
    
    config_dir1 = Path.home() / ".config" / "pybliometrics"
    config_dir2 = Path.home() / ".scopus"
    config_dir1.mkdir(parents=True, exist_ok=True)
    config_dir2.mkdir(parents=True, exist_ok=True)
    
    config_files = [config_dir1 / "config.ini", config_dir2 / "config.ini"]
    
    config_content = f"""[Authentication]
APIKey = {api_key}
InstToken = 

[Directories]
AbstractRetrieval = {{path}}/abstract_retrieval
AffiliationSearch = {{path}}/affiliation_search
AuthorRetrieval = {{path}}/author_retrieval
AuthorSearch = {{path}}/author_search
CitationOverview = {{path}}/citation_overview
ScopusSearch = {{path}}/scopus_search
SerialSearch = {{path}}/serial_search
SerialTitle = {{path}}/serial_title
SubjectClassifications = {{path}}/subject_classifications
PlumXMetrics = {{path}}/plumx_metrics

[Requests]
Timeout = 20
Retries = 5
"""
    for config_file in config_files:
        config_file.write_text(config_content)
    
    import pybliometrics
    try:
        pybliometrics.init()
    except Exception as e:
        pass # If it throws, at least we tried
    print("Configuración de pybliometrics inicializada automáticamente.")

def fetch_chunk(query, chunk_name):
    """
    Descarga una porción específica de datos. 
    Guarda resultados parciales para poder reanudar si hay fallas.
    """
    output_file = DATA_DIR / f"{chunk_name}.parquet"
    if output_file.exists():
        print(f"[-] Saltando {chunk_name}, ya existe en caché local.")
        return pd.read_parquet(output_file)
    
    print(f"[*] Descargando {chunk_name} (Query: {query})")
    try:
        from pybliometrics.scopus import ScopusSearch
        s = ScopusSearch(query, download=True, subscriber=False) # False para evitar errores 403 de vistas detalladas en cuentas estándar
        if not s.results:
            print("    -> 0 resultados.")
            return None
            
        df = pd.DataFrame(s.results)
        # Guardar checkpoint local
        df.to_parquet(output_file, index=False)
        print(f"    -> {len(df)} resultados guardados.")
        return df
    except Exception as e:
        print(f"[!] Error descargando {chunk_name}: {e}")
        return None

def download_by_asjc(asjc_code, start_year, end_year):
    """Descarga iterando por año para asegurar robustez."""
    all_dfs = []
    
    for year in range(start_year, end_year + 1):
        # Para ASJC grandes, dividimos el año en 2 semestres para evitar queries gigantes
        # Esto reduce el riesgo de timeouts y facilita retomar descargas fallidas.
        
        # Semestre 1: Jan - Jun
        q_h1 = f"SUBJAREA({asjc_code}) AND PUBYEAR = {year} AND (PUBDATETXT(* Jan *) OR PUBDATETXT(* Feb *) OR PUBDATETXT(* Mar *) OR PUBDATETXT(* Apr *) OR PUBDATETXT(* May *) OR PUBDATETXT(* Jun *))"
        df_h1 = fetch_chunk(q_h1, f"asjc_{asjc_code}_{year}_H1")
        if df_h1 is not None:
            all_dfs.append(df_h1)
            
        # Semestre 2: Jul - Dec
        q_h2 = f"SUBJAREA({asjc_code}) AND PUBYEAR = {year} AND (PUBDATETXT(* Jul *) OR PUBDATETXT(* Aug *) OR PUBDATETXT(* Sep *) OR PUBDATETXT(* Oct *) OR PUBDATETXT(* Nov *) OR PUBDATETXT(* Dec *))"
        df_h2 = fetch_chunk(q_h2, f"asjc_{asjc_code}_{year}_H2")
        if df_h2 is not None:
            all_dfs.append(df_h2)
            
        # Papers sin mes especificado en el año
        q_undef = f"SUBJAREA({asjc_code}) AND PUBYEAR = {year} AND NOT PUBDATETXT(* Jan *) AND NOT PUBDATETXT(* Feb *) AND NOT PUBDATETXT(* Mar *) AND NOT PUBDATETXT(* Apr *) AND NOT PUBDATETXT(* May *) AND NOT PUBDATETXT(* Jun *) AND NOT PUBDATETXT(* Jul *) AND NOT PUBDATETXT(* Aug *) AND NOT PUBDATETXT(* Sep *) AND NOT PUBDATETXT(* Oct *) AND NOT PUBDATETXT(* Nov *) AND NOT PUBDATETXT(* Dec *)"
        df_undef = fetch_chunk(q_undef, f"asjc_{asjc_code}_{year}_UNDEF")
        if df_undef is not None:
            all_dfs.append(df_undef)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Limpiar duplicados por si acaso
        final_df = final_df.drop_duplicates(subset=['eid'])
        output_path = DATA_DIR / f"full_asjc_{asjc_code}_{start_year}_{end_year}.parquet"
        final_df.to_parquet(output_path, index=False)
        print(f"\n[OK] Descarga completa. {len(final_df)} papers guardados en {output_path}")
        return final_df
    else:
        print("\n[!] No se descargaron datos.")
        return None

def download_custom_query(query, start_year, end_year, name_prefix=None):
    """Descarga iterando por año para un query de búsqueda libre."""
    all_dfs = []
    
    if name_prefix:
        import re
        # Limpiar caracteres especiales para el nombre del archivo
        query_id = re.sub(r'[^a-zA-Z0-9]', '_', name_prefix).lower()
        query_id = re.sub(r'_+', '_', query_id).strip('_')
    else:
        # Usamos un hash simple del query para el nombre de archivo
        import hashlib
        query_id = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
    
    for year in range(start_year, end_year + 1):
        # Query delimitado por año
        q_year = f"({query}) AND PUBYEAR = {year}"
        df_year = fetch_chunk(q_year, f"custom_{query_id}_{year}")
        if df_year is not None:
            all_dfs.append(df_year)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['eid'])
        output_path = DATA_DIR / f"full_custom_{query_id}_{start_year}_{end_year}.parquet"
        final_df.to_parquet(output_path, index=False)
        
        # Guardar metadata de la consulta
        meta_path = DATA_DIR / f"full_custom_{query_id}_{start_year}_{end_year}.json"
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": name_prefix,
                "query": query,
                "start_year": start_year,
                "end_year": end_year,
                "total_documents": len(final_df)
            }, f, indent=4)
            
        print(f"\n[OK] Descarga custom completa. {len(final_df)} papers guardados en {output_path}")
        return final_df
    else:
        print("\n[!] No se descargaron datos.")
        return None

def check_query_size(query):
    """Retorna el número de resultados para un query sin descargarlos."""
    try:
        from pybliometrics.scopus import ScopusSearch
        s = ScopusSearch(query, download=False, subscriber=False)
        print(s.get_results_size())
    except Exception as e:
        print(f"Error: {e}")

def download_by_cluster(cluster_id, start_year, end_year):
    """
    Nota: La búsqueda directa por Topic Cluster ID en ScopusSearch no está nativamente 
    soportada con un campo estándar (el campo TOPIC-CLUSTER existe pero no siempre
    funciona en cuentas estándar sin SciVal link).
    En este caso usaríamos el endpoint de SciVal directamente para obtener los Scopus IDs,
    pero por ahora implementaremos una búsqueda genérica u obtención de top papers.
    """
    print(f"[!] Implementación por Topic Cluster ID {cluster_id} requiere una llamada inicial a SciVal API.")
    print("    Por ahora, se recomienda usar el Modo ASJC.")
    # TODO: Implementar lookup de Scopus IDs via SciVal /scivalAI/topicCluster/
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scopus Downloader")
    parser.add_argument("--mode", choices=["asjc", "cluster", "custom", "check"], required=True, help="Modo de búsqueda")
    parser.add_argument("--id", help="Código ASJC (ej. 2740) o ID de Topic Cluster (requerido para mode asjc/cluster)")
    parser.add_argument("--query", help="Query crudo de Scopus (requerido para mode custom)")
    parser.add_argument("--name", help="Nombre descriptivo para la búsqueda (ej. 'UNAM Lab Nucl')")
    parser.add_argument("--years", default="2020-2025", help="Rango de años ej. 2020-2025")
    
    args = parser.parse_args()
    init_pybliometrics()
    
    try:
        start_year, end_year = map(int, args.years.split('-'))
    except:
        print("Error: El rango de años debe ser en formato YYYY-YYYY")
        sys.exit(1)
        
    if args.mode == 'asjc':
        if not args.id:
            print("Error: --id es requerido para el modo asjc")
            sys.exit(1)
        download_by_asjc(args.id, start_year, end_year)
    elif args.mode == 'cluster':
        if not args.id:
            print("Error: --id es requerido para el modo cluster")
            sys.exit(1)
        download_by_cluster(args.id, start_year, end_year)
    elif args.mode == 'custom':
        if not args.query:
            print("Error: --query es requerido para el modo custom")
            sys.exit(1)
        download_custom_query(args.query, start_year, end_year, args.name)
    elif args.mode == 'check':
        if not args.query:
            print("Error: --query es requerido")
            sys.exit(1)
        check_query_size(args.query)
