import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
BASE_PATH = Path(__file__).parent.parent
DATA_DIR = BASE_PATH / 'data'
CACHE_TEMAS_DIR = DATA_DIR / 'cache_temas'
CACHE_TEMAS_DIR.mkdir(parents=True, exist_ok=True)

# ClickHouse connection params
CH_HOST = os.environ.get('CH_HOST', 'localhost')
CH_PORT = int(os.environ.get('CH_PORT', 8123))
CH_USER = os.environ.get('CH_USER', 'default')
CH_PASSWORD = os.environ.get('CH_PASSWORD', '')
CH_DATABASE = os.environ.get('CH_DATABASE', 'rag')

# --- CLICKHOUSE CLIENT SETUP ---
def get_ch_client():
    import clickhouse_connect
    import uuid
    is_secure = (CH_PORT == 8124)
    session_id = f"topics_flat_{uuid.uuid4().hex}"
    
    try:
        return clickhouse_connect.get_client(
            host=CH_HOST,
            port=CH_PORT,
            username=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE,
            secure=is_secure,
            verify=False,
            session_id=session_id
        )
    except Exception as e:
        if is_secure:
            try:
                return clickhouse_connect.get_client(
                    host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD,
                    database=CH_DATABASE, secure=False, verify=False, session_id=session_id
                )
            except Exception:
                st.error(f"Error conectando a ClickHouse (incluso sin SSL): {e}")
        else:
            st.error(f"Error conectando a ClickHouse: {e}")
        return None

# --- JERARQUÍA ---
@st.cache_data
def get_hierarchy():
    """Obtiene la jerarquía Domain > Field > Subfield de works_flat."""
    try:
        client = get_ch_client()
        query = """
        SELECT DISTINCT 
            domain_name AS domain, 
            field_name AS field, 
            subfield_name AS subfield 
        FROM works_flat 
        ORDER BY domain, field, subfield
        """
        return client.query_df(query)
    except Exception as e:
        st.error(f"Error cargando jerarquía: {e}")
        return pd.DataFrame()

def compute_subfield_data_flat(subfield):
    """Calcula todas las métricas para un subcampo usando la tabla flat y optimización de Sandbox."""
    try:
        import sys, os as _os
        _src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'src')
        if _src not in sys.path: sys.path.insert(0, _src)
        from regions import GLOBAL_REGIONS
        
        client = get_ch_client()
        sub_clean = subfield.strip().replace(' ', '_').lower()
        
        status = st.status(f"Generando analítica multi-nivel para '{subfield}'...", expanded=True)
        status.write("Creando Sandbox en memoria...")
        client.command(f"DROP TEMPORARY TABLE IF EXISTS sandbox")
        client.command(f"""
            CREATE TEMPORARY TABLE sandbox AS
            SELECT 
                id, publication_year, fwci, percentile, is_top_10, is_top_1, 
                cited_by_count, oa_status, language, country_codes, 
                institution_ids, institution_types, source_id, sdgs, `type`, topic_id
            FROM works_flat 
            WHERE subfield_name = '{subfield}'
        """)
        status.write("Sandbox listo. Calculando métricas de impacto...")
        q_base = """
            SELECT 
                publication_year as year,
                topic_id as topic,
                count() as doc_count,
                sum(fwci) as fwci_sum,
                sum(percentile) as percentile_sum,
                sum(is_top_10) as top_10_sum,
                sum(is_top_1) as top_1_sum,
                sum(cited_by_count) as citations,
                -- OA Stats
                countIf(oa_status = 'gold') as gold_sum,
                countIf(oa_status = 'diamond') as diamond_sum,
                countIf(oa_status = 'green') as green_sum,
                countIf(oa_status = 'hybrid') as hybrid_sum,
                countIf(oa_status = 'bronze') as bronze_sum,
                countIf(oa_status = 'closed') as closed_sum,
                -- Language Stats
                countIf(language = 'en') as lang_en_sum,
                countIf(language = 'es') as lang_es_sum,
                countIf(language = 'pt') as lang_pt_sum
            FROM sandbox
            WHERE publication_year >= 1900
        """
        
        # 1a. Mundo
        df_mundo = client.query_df(f"{q_base} GROUP BY year, topic")
        df_mundo['entity_type'] = 'Mundo'
        df_mundo['entity_name'] = 'Mundo'
          # 1b. Regiones — hasAny sobre country_codes directamente, sin arrayJoin
        region_parts = []
        for reg, codes in GLOBAL_REGIONS.items():
            codes_str = ", ".join([f"'{c}'" for c in codes])
            region_parts.append(f"hasAny(country_codes, [{codes_str}]), '{reg}'")
        region_mapping_sql = f"multiIf({', '.join(region_parts)}, 'Other')"
        
        region_query = f"""
            SELECT year, topic, entity_name,
                   sum(doc_count) as doc_count, sum(fwci_sum) as fwci_sum, sum(percentile_sum) as percentile_sum,
                   sum(top_10_sum) as top_10_sum, sum(top_1_sum) as top_1_sum, sum(citations) as citations,
                   sum(gold_sum) as gold_sum, sum(diamond_sum) as diamond_sum, sum(green_sum) as green_sum,
                   sum(hybrid_sum) as hybrid_sum, sum(bronze_sum) as bronze_sum, sum(closed_sum) as closed_sum,
                   sum(lang_en_sum) as lang_en_sum, sum(lang_es_sum) as lang_es_sum, sum(lang_pt_sum) as lang_pt_sum
            FROM (
                SELECT publication_year as year, topic_id as topic,
                       {region_mapping_sql} as entity_name,
                       count() as doc_count, sum(fwci) as fwci_sum, sum(percentile) as percentile_sum,
                       sum(is_top_10) as top_10_sum, sum(is_top_1) as top_1_sum,
                       sum(cited_by_count) as citations,
                       countIf(oa_status = 'gold') as gold_sum, countIf(oa_status = 'diamond') as diamond_sum,
                       countIf(oa_status = 'green') as green_sum, countIf(oa_status = 'hybrid') as hybrid_sum,
                       countIf(oa_status = 'bronze') as bronze_sum, countIf(oa_status = 'closed') as closed_sum,
                       countIf(language = 'en') as lang_en_sum, countIf(language = 'es') as lang_es_sum,
                       countIf(language = 'pt') as lang_pt_sum
                FROM sandbox
                WHERE publication_year >= 1900
                GROUP BY year, topic, entity_name
            ) GROUP BY year, topic, entity_name
        """
        df_region = client.query_df(region_query)
        df_region['entity_type'] = 'Region'
        
        # 1c. Países — arrayJoin sin alias interno reutilizado
        all_codes = [f"'{c}'" for reg in GLOBAL_REGIONS.values() for c in reg]
        pais_query = f"""
            SELECT year, topic, entity_name,
                   sum(doc_count) as doc_count, sum(fwci_sum) as fwci_sum, sum(percentile_sum) as percentile_sum,
                   sum(top_10_sum) as top_10_sum, sum(top_1_sum) as top_1_sum, sum(citations) as citations,
                   sum(gold_sum) as gold_sum, sum(diamond_sum) as diamond_sum, sum(green_sum) as green_sum,
                   sum(hybrid_sum) as hybrid_sum, sum(bronze_sum) as bronze_sum, sum(closed_sum) as closed_sum,
                   sum(lang_en_sum) as lang_en_sum, sum(lang_es_sum) as lang_es_sum, sum(lang_pt_sum) as lang_pt_sum
            FROM (
                SELECT publication_year as year, topic_id as topic,
                       arrayJoin(country_codes) as entity_name,
                       count() as doc_count, sum(fwci) as fwci_sum, sum(percentile) as percentile_sum,
                       sum(is_top_10) as top_10_sum, sum(is_top_1) as top_1_sum,
                       sum(cited_by_count) as citations,
                       countIf(oa_status = 'gold') as gold_sum, countIf(oa_status = 'diamond') as diamond_sum,
                       countIf(oa_status = 'green') as green_sum, countIf(oa_status = 'hybrid') as hybrid_sum,
                       countIf(oa_status = 'bronze') as bronze_sum, countIf(oa_status = 'closed') as closed_sum,
                       countIf(language = 'en') as lang_en_sum, countIf(language = 'es') as lang_es_sum,
                       countIf(language = 'pt') as lang_pt_sum
                FROM sandbox
                WHERE publication_year >= 1900
                GROUP BY year, topic, entity_name
            )
            WHERE entity_name IN ({','.join(all_codes)})
            GROUP BY year, topic, entity_name
        """
        df_pais = client.query_df(pais_query)
        df_pais['entity_type'] = 'Country'
        
        df = pd.concat([df_mundo, df_region, df_pais], ignore_index=True)
        if df.empty: return False

        # Post-procesamiento
        for m in ['fwci', 'percentile', 'pct_top_10', 'pct_top_1']:
            is_pct = 'pct' in m
            sum_col = f"{m}_sum" if not is_pct else f"{m.replace('pct_','')}_sum"
            df[m] = np.where(df['doc_count'] > 0, (df[sum_col] / df['doc_count']) * (100 if is_pct else 1), 0)
        
        for col in ['gold', 'diamond', 'green', 'hybrid', 'bronze', 'closed']:
            df[f'pct_oa_{col}'] = np.where(df['doc_count'] > 0, (df[f'{col}_sum'] / df['doc_count']) * 100, 0)
        for lang in ['en', 'es', 'pt']:
            df[f'pct_lang_{lang}'] = np.where(df['doc_count'] > 0, (df[f'lang_{lang}_sum'] / df['doc_count']) * 100, 0)

        df.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_flat.parquet", index=False)
        
        # 4. Top Journals
        journal_query = """
        SELECT source_id as journal_id, any(S.display_name) as Revista, count() as articulos
        FROM sandbox AS W
        LEFT JOIN (SELECT id, argMax(display_name, updated_date) as display_name FROM sources GROUP BY id) AS S ON W.source_id = S.id
        WHERE publication_year >= 2021
        GROUP BY journal_id ORDER BY articulos DESC LIMIT 100
        """
        client.query_df(journal_query).to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_journals_flat.parquet", index=False)

        # 5. Colaboración Internacional
        collab_query = """
        SELECT c1 as country_a, c2 as country_b, count() as count
        FROM (SELECT arrayJoin(country_codes) as c1, country_codes as arr FROM sandbox WHERE length(country_codes) > 1 AND publication_year >= 1900)
        ARRAY JOIN arr as c2 WHERE c1 < c2
        GROUP BY country_a, country_b ORDER BY count DESC LIMIT 1000
        """
        client.query_df(collab_query).to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_collab_flat.parquet", index=False)

        status.write("Calculando analítica institucional...")
        q_inst_base = """
        SELECT publication_year as year, institution_id, if(institution_name = '', institution_id, institution_name) as institution_name,
               if(I.country_code = '', 'Unknown', I.country_code) as country_code, count() as doc_count,
               if(count() > 0, sum(fwci) / count(), 0) AS fwci, if(count() > 0, sum(percentile) / count(), 0) AS percentile,
               if(count() > 0, (sum(is_top_10) / count()) * 100, 0) AS pct_top_10, if(count() > 0, (sum(is_top_1) / count()) * 100, 0) AS pct_top_1,
               sum(cited_by_count) as citations, countIf(length(country_codes) > 1) as intl_collab, countIf(length(sdgs) > 0) as sdg_docs
        FROM (SELECT *, arrayJoin(institution_ids) as institution_id FROM sandbox WHERE publication_year >= 1900) AS W
        LEFT JOIN (SELECT id, any(display_name) as institution_name, any(country_code) as country_code FROM institutions GROUP BY id) AS I ON W.institution_id = I.id
        GROUP BY year, institution_id, institution_name, I.country_code
        """
        # Multi-segmento: Global Top 1000 + Mexico Top 500 + Regional Top 100
        inst_final_q = f"""
        SELECT * FROM (
            (SELECT *, 'Global' as segment FROM ({q_inst_base}) ORDER BY doc_count DESC LIMIT 1000)
            UNION DISTINCT
            (SELECT *, 'Mexico' as segment FROM ({q_inst_base}) WHERE country_code = 'MX' ORDER BY doc_count DESC LIMIT 500)
        )
        """
        df_inst = client.query_df(inst_final_q)
        
        mapping = {c: r for r, countries in GLOBAL_REGIONS.items() for c in countries}
        df_inst['region'] = df_inst['country_code'].map(mapping).fillna('Other')
        
        # Top 100 por Región (garantiza representatividad de LATAM, Africa, etc.)
        regional_dfs = []
        for reg_name, reg_codes in GLOBAL_REGIONS.items():
            df_reg = df_inst[df_inst['region'] == reg_name].copy()
            df_reg_agg = df_reg.groupby(['institution_id', 'institution_name', 'country_code', 'region']).agg({
                'doc_count': 'sum', 'fwci': 'mean', 'percentile': 'mean', 'pct_top_10': 'mean',
                'pct_top_1': 'mean', 'citations': 'sum', 'intl_collab': 'sum', 'sdg_docs': 'sum'
            }).reset_index()
            df_reg_agg['segment'] = 'Regional'
            regional_dfs.append(df_reg_agg.nlargest(100, 'doc_count'))
        
        if regional_dfs:
            df_regional = pd.concat(regional_dfs, ignore_index=True)
            df_inst = pd.concat([df_inst, df_regional], ignore_index=True).drop_duplicates(
                subset=['institution_id', 'country_code'], keep='first'
            )
        
        df_inst['award_docs'] = 0  # works_flat no tiene esta columna aun
        df_inst.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_inst_flat.parquet", index=False)

        status.write("Calculando tipos documentales y sectoriales...")
        type_q = """
        SELECT year, country_code, doc_type, count() as count
        FROM (SELECT DISTINCT id, publication_year as year, '' as country_code, `type` as doc_type FROM sandbox
              UNION ALL
              SELECT DISTINCT id, publication_year as year, arrayJoin(country_codes) as country_code, `type` as doc_type FROM sandbox)
        WHERE country_code != '' OR (country_code = '' AND id IS NOT NULL) GROUP BY year, country_code, doc_type
        """
        client.query_df(type_q).to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_types_flat.parquet", index=False)

        inst_type_q = """
        SELECT year, country_code, inst_type, count() as count
        FROM (SELECT DISTINCT id, publication_year as year, '' as country_code, arrayJoin(institution_types) as inst_type FROM sandbox
              UNION ALL
              SELECT DISTINCT id, publication_year as year, arrayJoin(country_codes) as country_code, arrayJoin(institution_types) as inst_type FROM sandbox)
        WHERE inst_type != '' GROUP BY year, country_code, inst_type
        """
        client.query_df(inst_type_q).to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_inst_types_flat.parquet", index=False)

        status.update(label=f"Analítica de '{subfield}' completada ✅", state="complete")
        return True
    except Exception as e:
        st.error(f"Error en ClickHouse (Flat Sandbox): {e}")
        return False
