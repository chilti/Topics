import pandas as pd
from datetime import datetime
import json
import numpy as np
import os

def _safe_join(arr, sep="; "):
    if arr is None: return ""
    if isinstance(arr, (list, np.ndarray)):
        # Filter None/nan
        clean = [str(x).strip() for x in arr if pd.notna(x) and str(x).strip()]
        return sep.join(clean)
    if pd.isna(arr): return ""
    return str(arr)

def dataframe_to_scopus_txt(df: pd.DataFrame) -> str:
    """
    Convierte un DataFrame de works_flat al formato de texto plano genérico de Scopus.
    """
    lines = []
    lines.append("Scopus")
    lines.append(f"EXPORT DATE: {datetime.now().strftime('%d %B %Y')}")
    lines.append("")
    
    for _, row in df.iterrows():
        # 1. Nombres cortos (Sarkis Y., D'Olivo J.C.)
        authors_raw = row.get('author_names', [])
        authors = _safe_join(authors_raw, "; ") if isinstance(authors_raw, (list, np.ndarray)) else ""
        
        # Generar nombres cortos burdamente (Ej. Youssef Sarkis -> Sarkis Y.)
        short_authors = []
        if isinstance(authors_raw, (list, np.ndarray)):
            for name in authors_raw:
                if pd.isna(name): continue
                parts = str(name).split()
                if len(parts) >= 2:
                    short_authors.append(f"{parts[-1]} {parts[0][0]}.")
                else:
                    short_authors.append(str(name))
        
        lines.append(", ".join(short_authors) if short_authors else "[No authors]")
        lines.append(f"AUTHOR FULL NAMES: {authors}")
        
        # Author IDs 
        auth_ids = row.get('author_ids', [])
        # Extraer solo el numero del OpenAlex ID (ej. https://openalex.org/A123 -> A123)
        clean_ids = []
        if isinstance(auth_ids, (list, np.ndarray)):
            clean_ids = [str(aid).split('/')[-1] for aid in auth_ids if pd.notna(aid)]
        lines.append("; ".join(clean_ids))
        
        # Title
        lines.append(str(row.get('title', '[No title]')))
        
        # Source info
        year = str(row.get('publication_year', ''))
        
        # Tratar de obtener Source Name de raw_data
        source_name = "[No source]"
        publisher = ""
        issn = ""
        raw_json = row.get('raw_data', '{}')
        if pd.notna(raw_json) and str(raw_json).strip() != "":
            try:
                data = json.loads(str(raw_json))
                loc = data.get('primary_location') or {}
                src = loc.get('source') or {}
                source_name = src.get('display_name', source_name)
                publisher = src.get('host_organization_name', "")
                issns = src.get('issn')
                if issns:
                    issn = issns[0] if isinstance(issns, list) else str(issns)
            except:
                pass
                
        volume = row.get('volume', '')
        issue = row.get('issue', '')
        first_page = row.get('first_page', '')
        last_page = row.get('last_page', '')
        
        vol_str = f", {volume}" if pd.notna(volume) and volume else ""
        iss_str = f" ({issue})" if pd.notna(issue) and issue else ""
        pp_str = ""
        if pd.notna(first_page) and first_page:
            pp_str = f", pp. {first_page}"
            if pd.notna(last_page) and last_page:
                pp_str += f"-{last_page}"
                
        cites = row.get('cited_by_count', 0)
        cites_str = f", Cited {int(cites)} times." if pd.notna(cites) else ""
        
        source_line = f"({year}) {source_name}{vol_str}{iss_str}{pp_str}{cites_str}"
        lines.append(source_line)
        
        # DOI & URLs
        doi = row.get('doi', '')
        if pd.notna(doi) and doi:
            lines.append(f"DOI: {str(doi).replace('https://doi.org/', '')}")
        
        openalex_id = row.get('id', '')
        lines.append(f"https://openalex.org/{openalex_id.split('/')[-1] if openalex_id else ''}")
        lines.append("")
        
        # Affiliations
        affils = _safe_join(row.get('institution_names', []), "; ")
        if affils:
            lines.append(f"AFFILIATIONS: {affils}")
            
        # Abstract
        abstract = row.get('abstract', '')
        if pd.notna(abstract) and str(abstract).strip() != "":
            lines.append(f"ABSTRACT: {abstract}")
            
        # Publisher & ISSN
        if publisher:
            lines.append(f"PUBLISHER: {publisher}")
        if issn:
            lines.append(f"ISSN: {issn}")
            
        # Language
        lang = row.get('language', '')
        if pd.notna(lang) and lang:
            lines.append(f"LANGUAGE OF ORIGINAL DOCUMENT: {lang}")
            
        # Type
        doc_type = row.get('type', '')
        if pd.notna(doc_type) and doc_type:
            lines.append(f"DOCUMENT TYPE: {str(doc_type).title()}")
            
        lines.append("PUBLICATION STAGE: Final")
        
        # Open Access
        oa = row.get('oa_status', '')
        if pd.notna(oa) and oa and str(oa) != "closed":
            lines.append(f"OPEN ACCESS: {str(oa).title()} Open Access")
            
        # Keywords
        keywords = row.get('keywords', [])
        if isinstance(keywords, (list, np.ndarray)) and len(keywords) > 0:
            clean_keywords = [str(k) for k in keywords if pd.notna(k)]
            if clean_keywords:
                lines.append(f"AUTHOR KEYWORDS: {'; '.join(clean_keywords)}")
                
        # Index Keywords
        concepts = row.get('concepts', [])
        if isinstance(concepts, (list, np.ndarray)) and len(concepts) > 0:
            clean_concepts = [str(c) for c in concepts if pd.notna(c)]
            if clean_concepts:
                lines.append(f"INDEX KEYWORDS: {'; '.join(clean_concepts)}")
                
        # References
        refs = row.get('referenced_works', [])
        if isinstance(refs, (list, np.ndarray)) and len(refs) > 0:
            clean_refs = [str(r).split('/')[-1] for r in refs if pd.notna(r)]
            if clean_refs:
                lines.append(f"REFERENCES: {'; '.join(clean_refs)}")
                
        lines.append("SOURCE: OpenAlex")
        lines.append("")
        
    return "\n".join(lines)

def raw_scopus_to_txt(df: pd.DataFrame) -> str:
    """
    Convierte el DataFrame original descargado de la API de Scopus al formato de texto de Scopus.
    """
    lines = []
    lines.append("Scopus")
    lines.append(f"EXPORT DATE: {datetime.now().strftime('%d %B %Y')}")
    lines.append("")
    
    for _, row in df.iterrows():
        authors_raw = row.get('author_names', "")
        
        if pd.notna(authors_raw) and str(authors_raw).strip() != "":
            # authors_raw usually looks like "Nellen L.; Sarkis Y."
            # First line is comma separated names: Nellen L., Sarkis Y.
            lines.append(str(authors_raw).replace(";", ",").replace(" ,", ","))
            lines.append(f"AUTHOR FULL NAMES: {authors_raw}")
        else:
            lines.append("[No authors]")
            lines.append("AUTHOR FULL NAMES: ")
            
        auth_ids = row.get('author_ids', "")
        lines.append(str(auth_ids) if pd.notna(auth_ids) else "")
        
        # Title
        lines.append(str(row.get('title', '[No title]')))
        
        # Source info
        coverDate = str(row.get('coverDate', ''))
        year = coverDate[:4] if coverDate else ''
        source_name = str(row.get('publicationName', '[No source]'))
        volume = str(row.get('volume', ''))
        issue = str(row.get('issueIdentifier', ''))
        pageRange = str(row.get('pageRange', ''))
        
        vol_str = f", {volume}" if volume and volume != 'nan' else ""
        iss_str = f" ({issue})" if issue and issue != 'nan' else ""
        pp_str = f", pp. {pageRange}" if pageRange and pageRange != 'nan' else ""
        
        cites = row.get('citedby_count', 0)
        cites_str = f", Cited {int(cites)} times." if pd.notna(cites) else ""
        
        source_line = f"({year}) {source_name}{vol_str}{iss_str}{pp_str}{cites_str}"
        lines.append(source_line)
        
        doi = str(row.get('doi', ''))
        if doi and doi != 'nan':
            lines.append(f"DOI: {doi}")
            lines.append(f"https://www.scopus.com/inward/record.uri?partnerID=HzOxMe3b&scp={str(row.get('eid',''))}&origin=inward")
        else:
            lines.append("")
        lines.append("")
        
        affils = str(row.get('affilname', ''))
        if affils and affils != 'nan':
            lines.append(f"AFFILIATIONS: {affils.replace(';', '; ')}")
            
        abstract = str(row.get('description', ''))
        if abstract and abstract != 'nan':
            lines.append(f"ABSTRACT: {abstract}")
            
        issn = str(row.get('issn', ''))
        if issn and issn != 'nan':
            lines.append(f"ISSN: {issn}")
            
        doc_type = str(row.get('subtypeDescription', ''))
        if doc_type and doc_type != 'nan':
            lines.append(f"DOCUMENT TYPE: {doc_type}")
            
        lines.append("PUBLICATION STAGE: Final")
        
        oa = row.get('openaccess', '')
        if pd.notna(oa) and str(oa) in ['1', 'True', 'true', '1.0']:
            lines.append("OPEN ACCESS: All Open Access")
            
        lines.append("SOURCE: Scopus")
        lines.append("")
        
    return "\n".join(lines)

def fetch_export_data(engine, context_id, entity_name, period_mode):
    """
    Obtiene los datos crudos para exportar.
    - engine: "OpenAlex" o "Scopus"
    - context_id: Nombre del subfield (si OpenAlex) o ruta al parquet (si Scopus)
    - entity_name: "Mundo", "México", "Latinoamérica y Caribe", etc.
    - period_mode: String indicando si es últimos 5 años o periodo completo.
    """
    from regions import GLOBAL_REGIONS
    
    # Filtro temporal
    is_recent = "5 años" in period_mode
    
    if engine == "Scopus":
        import pandas as pd
        if not os.path.exists(context_id):
            return None
            
        df = pd.read_parquet(context_id)
        
        if is_recent:
            df = df[df['publication_year'] >= 2021]
            
        if entity_name != "Mundo":
            if entity_name == "México":
                df = df[df['country_codes'].apply(lambda x: 'MX' in x if isinstance(x, (list, np.ndarray)) else False)]
            elif entity_name in GLOBAL_REGIONS:
                region_countries = set(GLOBAL_REGIONS[entity_name])
                df = df[df['country_codes'].apply(lambda x: any(c in region_countries for c in x) if isinstance(x, (list, np.ndarray)) else False)]
                
        return df
        
    elif engine == "OpenAlex":
        # Extraer de ClickHouse
        from pipeline_topic.compute_metrics_flat import get_ch_client
        client = get_ch_client(silent=True)
        if not client:
            return None
            
        year_filter = "publication_year >= 2021 AND publication_year <= 2025" if is_recent else "publication_year >= 1900"
        
        entity_filter = "1=1"
        if entity_name == "México":
            entity_filter = "has(country_codes, 'MX')"
        elif entity_name in GLOBAL_REGIONS:
            countries = GLOBAL_REGIONS[entity_name]
            c_list = ", ".join([f"'{c}'" for c in countries])
            entity_filter = f"hasAny(country_codes, [{c_list}])"
            
        query = f"""
            SELECT * 
            FROM works_flat
            WHERE subfield = '{context_id}'
              AND {year_filter}
              AND {entity_filter}
        """
        try:
            df = client.query_df(query)
            # drop duplicates for replacingMergeTree
            if 'id' in df.columns:
                df = df.drop_duplicates(subset=['id'], keep='last')
            return df
        except Exception as e:
            print(f"Error extracting from clickhouse: {e}")
            return None
            
    return None
