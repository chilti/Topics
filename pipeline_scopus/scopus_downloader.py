import os
import sys
import json
import time
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import re

def split_query_by_parenthesis_depth(query_str, split_depths=(1, 2)):
    """Splits the query by 'AND NOT' when parenthesis depth is in split_depths."""
    parts = []
    current_part = []
    depth = 0
    i = 0
    n = len(query_str)
    
    while i < n:
        char = query_str[i]
        if char == '(':
            depth += 1
            current_part.append(char)
            i += 1
        elif char == ')':
            depth -= 1
            current_part.append(char)
            i += 1
        elif depth in split_depths and query_str[i:i+7].upper() in ('AND NOT', 'AND\nNOT', 'AND\rNOT'):
            parts.append("".join(current_part).strip())
            current_part = []
            i += 7
            while i < n and query_str[i].isspace():
                i += 1
        else:
            current_part.append(char)
            i += 1
            
    if current_part:
        parts.append("".join(current_part).strip())
        
    return parts

def extract_field_contents(query_str, field_name):
    """Finds all occurrences of field_name(...) in query_str, handling nested parentheses."""
    contents = []
    pattern = rf'\b{field_name}\s*\('
    for match in re.finditer(pattern, query_str, re.IGNORECASE):
        start_idx = match.end()
        depth = 1
        curr_idx = start_idx
        while depth > 0 and curr_idx < len(query_str):
            char = query_str[curr_idx]
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            curr_idx += 1
        
        if depth == 0:
            content = query_str[start_idx:curr_idx-1].strip()
            contents.append(content)
            
    return contents

def split_by_top_level_or(content):
    """Splits the content of a field by 'OR' at top level (depth 0)."""
    tokens = []
    current_token = []
    depth = 0
    i = 0
    n = len(content)
    
    while i < n:
        char = content[i]
        if char == '(':
            depth += 1
            current_token.append(char)
            i += 1
        elif char == ')':
            depth -= 1
            current_token.append(char)
            i += 1
        elif depth == 0 and content[i:i+4].upper() in (' OR ', '\nOR ', '\rOR '):
            tokens.append("".join(current_token).strip())
            current_token = []
            i += 3
        elif depth == 0 and content[i:i+3].upper() == 'OR\n':
            tokens.append("".join(current_token).strip())
            current_token = []
            i += 2
        else:
            current_token.append(char)
            i += 1
            
    if current_token:
        tokens.append("".join(current_token).strip())
        
    clean_tokens = []
    for t in tokens:
        t = t.strip()
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1]
        if t:
            clean_tokens.append(t)
            
    return clean_tokens

def extract_terms_robust(query, field_name):
    contents = extract_field_contents(query, field_name)
    terms = []
    for c in contents:
        terms.extend(split_by_top_level_or(c))
    return terms

def format_scopus_term(t):
    t = t.strip()
    if t.startswith('(') and t.endswith(')'):
        return t
    if t.startswith('"') and t.endswith('"'):
        return t
    if ' ' in t and not any(op in t.upper() for op in [' W/', ' PRE/', ' AND ', ' OR ', ' NOT ']):
        return f'"{t}"'
    return t

def parse_scopus_query(query_str):
    """
    Parses a Scopus query into:
      - pos_terms: List of positive keywords inside TITLE-ABS-KEY, TITLE, ABS, KEY
      - subjterms: List of SUBJTERMS or SUBJAREA inclusion codes
      - exclusions: Dictionary of exclusion list configurations to be applied locally.
    """
    # 1. Split query into top-level inclusion and exclusion blocks
    top_parts = split_query_by_parenthesis_depth(query_str, split_depths=(1, 2))
    inclusion_part = top_parts[0]
    exclusion_blocks = top_parts[1:]
    
    # 2. Extract positive terms
    pos_terms = []
    for field in ['TITLE-ABS-KEY', 'TITLE', 'ABS', 'KEY']:
        pos_terms.extend(extract_terms_robust(inclusion_part, field))
    pos_terms = list(dict.fromkeys(pos_terms))
    
    subjterms = re.findall(r'SUBJTERMS\s*\(\s*(\d+)\s*\)', inclusion_part, re.IGNORECASE)
    subjareas = re.findall(r'SUBJAREA\s*\(\s*([a-zA-Z]+)\s*\)', inclusion_part, re.IGNORECASE)
    
    all_inclusion_codes = []
    for code in subjterms:
        all_inclusion_codes.append(f"SUBJTERMS({code})")
    for code in subjareas:
        all_inclusion_codes.append(f"SUBJAREA({code})")
        
    # 3. Parse exclusions and re-inclusions
    # We will format this as a list of dictionaries, where each dict is:
    # { 'exclude_terms': [...], 'reinclude_terms': [...], 'exclude_vete': bool }
    exclusions = []
    for block in exclusion_blocks:
        inner_parts = split_query_by_parenthesis_depth(block, split_depths=(1, 2))
        
        # Part 0 of block is the exclusion criteria
        excl_part = inner_parts[0]
        # Part 1 of block is the re-inclusion criteria (optional)
        reincl_part = inner_parts[1] if len(inner_parts) > 1 else ""
        
        # Check for SUBJAREA(VETE) in exclusion
        exclude_vete = bool(re.search(r'SUBJAREA\s*\(\s*VETE\s*\)', excl_part, re.IGNORECASE))
        
        # Extract terms
        excl_terms = []
        for field in ['TITLE-ABS-KEY', 'TITLE', 'ABS', 'KEY']:
            excl_terms.extend(extract_terms_robust(excl_part, field))
        excl_terms = list(dict.fromkeys(excl_terms))
        
        reincl_terms = []
        if reincl_part:
            for field in ['TITLE-ABS-KEY', 'TITLE', 'ABS', 'KEY']:
                reincl_terms.extend(extract_terms_robust(reincl_part, field))
            reincl_terms = list(dict.fromkeys(reincl_terms))
            
        exclusions.append({
            'exclude_terms': excl_terms,
            'reinclude_terms': reincl_terms,
            'exclude_vete': exclude_vete
        })
        
    return {
        'pos_terms': pos_terms,
        'inclusion_codes': all_inclusion_codes,
        'exclusions': exclusions
    }


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
        error_msg = str(e)
        m = _re.search(r'Found ([\d,]+) matches', error_msg)
        if m:
            return int(m.group(1).replace(',', ''))
        elif "Exceeds the maximum number" in error_msg:
            # Fallback to direct HTTP request
            import requests
            import os
            from urllib.parse import quote
            api_key = os.environ.get("PYBLIOMETRICS_API_KEY", "72952ef45b534fa6e6f386ad415f89d9")
            headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
            url = f"https://api.elsevier.com/content/search/scopus?query={quote(query)}&count=1"
            try:
                res = requests.get(url, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    total = data.get('search-results', {}).get('opensearch:totalResults')
                    if total is not None:
                        return int(total)
            except Exception as inner_e:
                print(f"[!] Error en fallback HTTP de check_size: {inner_e}")
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

def download_by_date_bisection(base_query, chunk_prefix, start_date, end_date, all_dfs):
    """
    Divide recursivamente un rango de fechas de carga (LOAD-DATE) a la mitad 
    hasta que el número de resultados sea <= 5000.
    """
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    q_ld = f"({base_query}) AND LOAD-DATE AFT {start_str} AND LOAD-DATE BEF {end_str}"
    chunk_name = f"{chunk_prefix}_LD_{start_str}_{end_str}"
    
    size_ld = check_size(q_ld)
    if size_ld == 0:
        return
        
    if size_ld <= 5000:
        df_ld = fetch_chunk(q_ld, chunk_name)
        if df_ld is not None:
            all_dfs.append(df_ld)
    else:
        delta_days = (end_date - start_date).days
        if delta_days <= 2:
            print(f"[!!] Rango {chunk_name} no puede dividirse más y excede 5000 ({size_ld}). Descargando...")
            df_ld = fetch_chunk(q_ld, chunk_name)
            if df_ld is not None:
                all_dfs.append(df_ld)
            return
            
        mid_days = delta_days // 2
        mid_date = start_date + timedelta(days=mid_days)
        
        # Lado izquierdo
        download_by_date_bisection(base_query, chunk_prefix, start_date, mid_date + timedelta(days=1), all_dfs)
        # Lado derecho
        download_by_date_bisection(base_query, chunk_prefix, mid_date, end_date, all_dfs)

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
    """Descarga iterando por año para un query de búsqueda libre, dividiendo en chunks si es necesario."""
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
        
    # 1. Analizar consulta Scopus
    print(f"[*] Analizando consulta Scopus para descarga segmentada...")
    parsed = parse_scopus_query(query)
    pos_terms = parsed['pos_terms']
    inclusion_codes = parsed['inclusion_codes']
    exclusions = parsed['exclusions']
    
    print(f"    - Códigos de inclusión directa: {inclusion_codes}")
    print(f"    - Términos de búsqueda positivos: {len(pos_terms)}")
    print(f"    - Bloques de exclusión: {len(exclusions)}")
    
    # Check if any exclusions require filtering VETE at the API level
    api_exclusions = ""
    for excl in exclusions:
        if excl.get('exclude_vete'):
            api_exclusions = " AND NOT SUBJAREA(VETE)"
            print("    [!] Detectado filtro de Veterinaria (VETE). Se aplicará directamente en la API de Scopus.")
            break
            
    # 2. Generar sub-consultas (Chunks)
    sub_queries = []
    # Primero agregamos los códigos directos (e.g. SUBJTERMS(2740))
    for code in inclusion_codes:
        sub_queries.append((code, f"code_{re.sub(r'[^a-zA-Z0-9]', '', code).lower()}"))
        
    # Agrupamos los términos de búsqueda en bloques de 30 para evitar URLs demasiado largas
    chunk_size = 30
    if pos_terms:
        formatted_terms = [format_scopus_term(t) for t in pos_terms]
        for idx, i in enumerate(range(0, len(formatted_terms), chunk_size)):
            chunk = formatted_terms[i:i + chunk_size]
            chunk_q = " OR ".join(chunk)
            sub_queries.append((f"TITLE-ABS-KEY({chunk_q})", f"chunk_{idx+1}"))
            
    # Fallback si no se pudieron extraer partes (por ejemplo, si la estructura de la consulta era atípica)
    if not sub_queries:
        print("    [⚠️] No se pudieron extraer sub-consultas estructuradas. Se usará la consulta original completa.")
        sub_queries = [(query, "raw")]
        
    # 3. Ejecutar descargas por año y por chunk
    print(f"[*] Total de segmentos a descargar por año: {len(sub_queries)}")
    for year in range(start_year, end_year + 1):
        print(f"\n==================== PROCESANDO AÑO {year} ====================")
        for sub_q, chunk_id in sub_queries:
            q_year = f"({sub_q}){api_exclusions} AND PUBYEAR = {year}"
            chunk_filename = f"custom_{query_id}_{year}_{chunk_id}"
            
            # Checar el volumen de esta combinación
            size = check_size(q_year)
            if size == 0:
                continue
                
            print(f"[*] Segmento '{chunk_id}' en {year}: {size} resultados.")
            
            if size <= 5000:
                # Seguro descargar el año entero
                df_year = fetch_chunk(q_year, chunk_filename)
                if df_year is not None:
                    all_dfs.append(df_year)
            else:
                print(f"[*] Segmento '{chunk_id}' en {year} tiene {size} resultados (>5000). Dividiendo por meses...")
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                
                # Descargar mes por mes
                for month in months:
                    q_month = f"({sub_q}){api_exclusions} AND PUBYEAR = {year} AND PUBDATETXT(* {month} *)"
                    df_m = fetch_chunk(q_month, f"{chunk_filename}_{month}")
                    if df_m is not None:
                        all_dfs.append(df_m)
                        
                # Descargar documentos que no especifican mes
                q_undef = f"({sub_q}){api_exclusions} AND PUBYEAR = {year} AND NOT PUBDATETXT(* Jan *) AND NOT PUBDATETXT(* Feb *) AND NOT PUBDATETXT(* Mar *) AND NOT PUBDATETXT(* Apr *) AND NOT PUBDATETXT(* May *) AND NOT PUBDATETXT(* Jun *) AND NOT PUBDATETXT(* Jul *) AND NOT PUBDATETXT(* Aug *) AND NOT PUBDATETXT(* Sep *) AND NOT PUBDATETXT(* Oct *) AND NOT PUBDATETXT(* Nov *) AND NOT PUBDATETXT(* Dec *)"
                
                size_undef = check_size(q_undef)
                
                if size_undef <= 5000:
                    df_undef = fetch_chunk(q_undef, f"{chunk_filename}_UNDEF")
                    if df_undef is not None:
                        all_dfs.append(df_undef)
                else:
                    print(f"[*] Segmento '{chunk_id}' en {year} UNDEF tiene {size_undef} resultados (>5000). Dividiendo por DOCTYPE...")
                    doctypes = ['ar', 're', 'cp', 'bk', 'ch', 'ed', 'sh', 'le', 'no', 'er', 'cr']
                    seen_doctypes = []
                    for dt in doctypes:
                        q_dt = f"({q_undef}) AND DOCTYPE({dt})"
                        chunk_name = f"{chunk_filename}_UNDEF_{dt}"
                        size_dt = check_size(q_dt)
                        seen_doctypes.append(dt)
                        
                        if size_dt <= 5000:
                            df_dt = fetch_chunk(q_dt, chunk_name)
                            if df_dt is not None:
                                all_dfs.append(df_dt)
                        elif size_dt > 0:
                            # Subdividir por AREA DE CONOCIMIENTO (SUBJAREA)
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
                                        print(f"[!] {area_chunk_name} sigue teniendo {size_area} resultados. Subdividiendo recursivamente por LOAD-DATE...")
                                        start_date = datetime(year, 1, 1)
                                        end_date = datetime.now() + timedelta(days=365*2)
                                        download_by_date_bisection(q_area, area_chunk_name, start_date, end_date, all_dfs)
                            # Complemento de áreas
                            not_areas = " AND ".join([f"NOT SUBJAREA({a})" for a in seen_areas])
                            if not_areas:
                                q_other = f"({q_dt}) AND {not_areas}"
                                other_chunk_name = f"{chunk_name}_area_other"
                                size_other = check_size(q_other)
                                if size_other > 0:
                                    if size_other <= 5000:
                                        df_area_other = fetch_chunk(q_other, other_chunk_name)
                                        if df_area_other is not None:
                                            all_dfs.append(df_area_other)
                                    else:
                                        print(f"[!] {other_chunk_name} tiene {size_other} resultados. Subdividiendo recursivamente por LOAD-DATE...")
                                        start_date = datetime(year, 1, 1)
                                        end_date = datetime.now() + timedelta(days=365*2)
                                        download_by_date_bisection(q_other, other_chunk_name, start_date, end_date, all_dfs)
                    
                    # Complemento de DOCTYPEs
                    not_dt = " AND ".join([f"NOT DOCTYPE({dt})" for dt in doctypes])
                    q_not_dt = f"({q_undef}) AND {not_dt}"
                    other_dt_chunk = f"{chunk_filename}_UNDEF_OTHER"
                    size_not_dt = check_size(q_not_dt)
                    if size_not_dt > 0:
                        if size_not_dt <= 5000:
                            df_not_dt = fetch_chunk(q_not_dt, other_dt_chunk)
                            if df_not_dt is not None:
                                all_dfs.append(df_not_dt)
                        else:
                            print(f"[!] {other_dt_chunk} tiene {size_not_dt} resultados. Subdividiendo recursivamente por LOAD-DATE...")
                            start_date = datetime(year, 1, 1)
                            end_date = datetime.now() + timedelta(days=365*2)
                            download_by_date_bisection(q_not_dt, other_dt_chunk, start_date, end_date, all_dfs)
                            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['eid'])
        output_path = DATA_DIR / f"full_custom_{query_id}_{start_year}_{end_year}.parquet"
        final_df.to_parquet(output_path, index=False)
        
        # Guardar metadata de la consulta incluyendo las exclusiones para scopus_processor.py
        meta_path = DATA_DIR / f"full_custom_{query_id}_{start_year}_{end_year}.json"
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": name_prefix,
                "query": query,
                "start_year": start_year,
                "end_year": end_year,
                "total_documents": len(final_df),
                "exclusions": exclusions
            }, f, indent=4, ensure_ascii=False)
            
        print(f"\n[OK] Descarga custom completa. {len(final_df)} papers guardados en {output_path}")
        
        # Procesamiento automático post-descarga
        try:
            print("[*] Iniciando cruce con OpenAlex y aplicación de exclusiones locales...")
            sys.path.append(str(BASE_PATH))
            from pipeline_scopus.scopus_processor import procesar_scopus
            procesar_scopus(output_path)
            print("[OK] Cruce, filtrado y procesamiento completo.")
        except Exception as e:
            print(f"[!] Error durante el cruce/procesamiento: {e}")
            
        return final_df
    else:
        print("\n[!] No se descargaron datos.")
        return None

def check_query_size(query):
    """Retorna el número de resultados para un query sin descargarlos."""
    print(check_size(query))

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
