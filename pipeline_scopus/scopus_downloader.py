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
BASE_PATH = Path(__file__).parent.parent
DATA_DIR = BASE_PATH / "data" / "cache_scopus" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Áreas de conocimiento de Scopus (ASJC) en orden de probable volumen
SCOPUS_SUBJECT_AREAS = [
    ('MEDI', 'Medicine'),
    ('SOCI', 'Social Sciences'),
    ('PSYC', 'Psychology'),
    ('NURS', 'Nursing'),
    ('HEAL', 'Health Professions'),
    ('BIOC', 'Biochemistry'),
    ('ENVI', 'Environmental Science'),
    ('AGRI', 'Agricultural and Biological Sciences'),
    ('COMP', 'Computer Science'),
    ('ENGI', 'Engineering'),
    ('BUSI', 'Business'),
    ('ECON', 'Economics'),
    ('MATH', 'Mathematics'),
    ('NEUR', 'Neuroscience'),
    ('PHAR', 'Pharmacology'),
    ('ARTS', 'Arts and Humanities'),
    ('PHYS', 'Physics and Astronomy'),
    ('CHEM', 'Chemistry'),
    ('EART', 'Earth and Planetary Sciences'),
    ('DECI', 'Decision Sciences'),
    ('IMMU', 'Immunology'),
    ('MATE', 'Materials Science'),
    ('ENER', 'Energy'),
    ('VETE', 'Veterinary'),
    ('DENT', 'Dentistry'),
    ('MULT', 'Multidisciplinary'),
]

def check_size(query):
    """Retorna el numero de resultados de un query sin descargarlo."""
    try:
        from pybliometrics.scopus import ScopusSearch
        s = ScopusSearch(query, download=False, subscriber=False)
        return s.get_results_size()
    except Exception as e:
        import re as _re
        m = _re.search(r'Found ([\d,]+) matches', str(e))
        if m:
            return int(m.group(1).replace(',', ''))
        return 0

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
        
        # Procesamiento automático post-descarga
        try:
            print("[*] Iniciando cruce con OpenAlex...")
            sys.path.append(str(BASE_PATH))
            from pipeline_scopus.scopus_processor import procesar_scopus
            procesar_scopus(output_path)
            print("[OK] Cruce y procesamiento completo.")
        except Exception as e:
            print(f"[!] Error durante el cruce con OpenAlex: {e}")
            
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
        
        # 1. Checar el volumen del año
        size = 0
        try:
            from pybliometrics.scopus import ScopusSearch
            s_check = ScopusSearch(q_year, download=False, subscriber=False)
            size = s_check.get_results_size()
        except Exception as e:
            import re
            m = re.search(r'Found ([\d,]+) matches', str(e))
            if m:
                size = int(m.group(1).replace(',', ''))
            
        if size <= 5000:
            # Seguro descargar el año entero
            df_year = fetch_chunk(q_year, f"custom_{query_id}_{year}")
            if df_year is not None:
                all_dfs.append(df_year)
        else:
            print(f"[*] Año {year} tiene {size} resultados (>5000). Dividiendo por meses...")
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            # Descargar mes por mes
            for month in months:
                q_month = f"({query}) AND PUBYEAR = {year} AND PUBDATETXT(* {month} *)"
                df_m = fetch_chunk(q_month, f"custom_{query_id}_{year}_{month}")
                if df_m is not None:
                    all_dfs.append(df_m)
                    
            # Descargar documentos que no especifican mes
            q_undef = f"({query}) AND PUBYEAR = {year} AND NOT PUBDATETXT(* Jan *) AND NOT PUBDATETXT(* Feb *) AND NOT PUBDATETXT(* Mar *) AND NOT PUBDATETXT(* Apr *) AND NOT PUBDATETXT(* May *) AND NOT PUBDATETXT(* Jun *) AND NOT PUBDATETXT(* Jul *) AND NOT PUBDATETXT(* Aug *) AND NOT PUBDATETXT(* Sep *) AND NOT PUBDATETXT(* Oct *) AND NOT PUBDATETXT(* Nov *) AND NOT PUBDATETXT(* Dec *)"
            
            size_undef = 0
            try:
                from pybliometrics.scopus import ScopusSearch
                s_undef = ScopusSearch(q_undef, download=False, subscriber=False)
                size_undef = s_undef.get_results_size()
            except Exception as e:
                import re
                m = re.search(r'Found ([\d,]+) matches', str(e))
                if m:
                    size_undef = int(m.group(1).replace(',', ''))
            
            if size_undef <= 5000:
                df_undef = fetch_chunk(q_undef, f"custom_{query_id}_{year}_UNDEF")
                if df_undef is not None:
                    all_dfs.append(df_undef)
            else:
                print(f"[*] Año {year} UNDEF tiene {size_undef} resultados (>5000). Dividiendo por DOCTYPE...")
                doctypes = ['ar', 're', 'cp', 'bk', 'ch', 'ed', 'sh', 'le', 'no', 'er', 'cr']
                seen_doctypes = []
                for dt in doctypes:
                    q_dt = f"({q_undef}) AND DOCTYPE({dt})"
                    chunk_name = f"custom_{query_id}_{year}_UNDEF_{dt}"
                    size_dt = check_size(q_dt)
                    seen_doctypes.append(dt)
                    
                    if size_dt <= 5000:
                        df_dt = fetch_chunk(q_dt, chunk_name)
                        if df_dt is not None:
                            all_dfs.append(df_dt)
                    elif size_dt > 0:
                        # Tercer nivel: subdividir por AREA DE CONOCIMIENTO (SUBJAREA)
                        print(f"[*] {chunk_name} tiene {size_dt} resultados (>5000). Subdividiendo por SUBJAREA...")
                        seen_areas = []
                        for area_code, area_name in SCOPUS_SUBJECT_AREAS:
                            q_area = f"({q_dt}) AND SUBJAREA({area_code})"
                            area_chunk_name = f"{chunk_name}_{area_code.lower()}"
                            size_area = check_size(q_area)
                            seen_areas.append(area_code)
                            if size_area > 0:
                                if size_area <= 5000:
                                    df_area = fetch_chunk(q_area, area_chunk_name)
                                    if df_area is not None:
                                        all_dfs.append(df_area)
                                else:
                                    print(f"[!] {area_chunk_name} sigue teniendo {size_area} resultados. Se descargará en bloques de 5000 máximo.")
                                    # Último recurso: intentar descargar de todos modos (pybliometrics cortara en 5000)
                                    df_area = fetch_chunk(q_area, area_chunk_name)
                                    if df_area is not None:
                                        all_dfs.append(df_area)
                        # Complemento de áreas
                        not_areas = " AND ".join([f"NOT SUBJAREA({a})" for a in seen_areas])
                        if not_areas:
                            df_area_other = fetch_chunk(f"({q_dt}) AND {not_areas}", f"{chunk_name}_area_other")
                            if df_area_other is not None:
                                all_dfs.append(df_area_other)
                
                # Complemento de DOCTYPEs
                not_dt = " AND ".join([f"NOT DOCTYPE({dt})" for dt in doctypes])
                q_not_dt = f"({q_undef}) AND {not_dt}"
                df_not_dt = fetch_chunk(q_not_dt, f"custom_{query_id}_{year}_UNDEF_OTHER")
                if df_not_dt is not None:
                    all_dfs.append(df_not_dt)
            
            
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
        
        # Procesamiento automático post-descarga
        try:
            print("[*] Iniciando cruce con OpenAlex...")
            sys.path.append(str(BASE_PATH))
            from pipeline_scopus.scopus_processor import procesar_scopus
            procesar_scopus(output_path)
            print("[OK] Cruce y procesamiento completo.")
        except Exception as e:
            print(f"[!] Error durante el cruce con OpenAlex: {e}")
            
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
        import re
        error_msg = str(e)
        # Pybliometrics throws an error if > 5000 even when download=False
        # e.g., "Found 54,485 matches. The query fails to return more than 5000 entries..."
        m = re.search(r'Found ([\d,]+) matches', error_msg)
        if m:
            num_str = m.group(1).replace(',', '')
            print(num_str)
        else:
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
