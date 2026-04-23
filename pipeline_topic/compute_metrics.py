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
    try:
        import clickhouse_connect
        return clickhouse_connect.get_client(
            host=CH_HOST,
            port=CH_PORT,
            username=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE
        )
    except Exception as e:
        st.error(f"Error conectando a ClickHouse: {e}")
        return None

# --- JERARQUÍA ---
@st.cache_data
def get_hierarchy():
    """Obtiene la jerarquía Domain > Field > Subfield, usando caché local o ClickHouse."""
    hierarchy_path = DATA_DIR / 'cache' / 'topic_hierarchy.parquet'
    
    # 1. Intentar cargar desde el archivo específico de jerarquía
    if hierarchy_path.exists():
        return pd.read_parquet(hierarchy_path)
    
    # 2. Intentar extraer de sunburst_metrics_latam (si existe)
    sunburst_path = DATA_DIR / 'cache' / 'sunburst_metrics_latam.parquet'
    df_hier = None
    
    if sunburst_path.exists():
        df_hier = pd.read_parquet(sunburst_path)
        df_hier = df_hier[df_hier['level'].isin(['domain', 'field', 'subfield'])]
        df_hier = df_hier[['domain', 'field', 'subfield']].drop_duplicates()
        df_hier = df_hier.replace('ALL', np.nan)
    else:
        # 3. Consultar ClickHouse como último recurso
        client = get_ch_client()
        if not client: return None
        
        query = """
        SELECT DISTINCT
            domain,
            field,
            subfield
        FROM works
        WHERE domain != ''
        """
        try:
            df_hier = client.query_df(query)
        except Exception as e:
            st.error(f"Error cargando jerarquía de ClickHouse: {e}")
            return None

    # Guardar para la próxima vez si logramos obtenerla
    if df_hier is not None and not df_hier.empty:
        df_hier.to_parquet(hierarchy_path, index=False)
        return df_hier
    
    return None

# --- CALCULATION LOGIC ---
def compute_subfield_data(subfield):
    """Ejecuta consulta OLAP en ClickHouse. Realiza agregaciones multi-nivel (País, Región, Mundo)."""
    from regions import GLOBAL_REGIONS
    client = get_ch_client()
    if not client: return False
    
    # 1. Generar multiIf para mapeo regional en SQL
    region_cases = []
    for region, countries in GLOBAL_REGIONS.items():
        # Escapar nombres de regiones y códigos
        c_list = ", ".join([f"'{c}'" for c in countries])
        region_cases.append(f"country_code IN ({c_list}), '{region}'")
    
    region_mapping_sql = f"multiIf({', '.join(region_cases)}, 'Other')"
    
    # 2. Consultas Secuenciales (Más robusto que GROUPING SETS para este caso)
    try:
        with st.status(f"Generando analítica multi-nivel para '{subfield}'..."):
            # 2a. Mundo (Por año y tópico)
            q_mundo = f"""
            SELECT 
                year, topic, 
                sum(doc_count) as doc_count, sum(fwci_sum) as fwci_sum, sum(percentile_sum) as percentile_sum,
                sum(top_10_sum) as top_10_sum, sum(top_1_sum) as top_1_sum,
                sum(gold_count) as gold_sum, sum(diamond_count) as diamond_sum, sum(green_count) as green_sum,
                sum(hybrid_count) as hybrid_sum, sum(bronze_count) as bronze_sum, sum(closed_count) as closed_sum,
                sum(lang_en) as lang_en_sum, sum(lang_es) as lang_es_sum, sum(lang_pt) as lang_pt_sum
            FROM summing_subfield_metrics WHERE subfield = '{subfield}' AND year >= 1900
            GROUP BY year, topic
            """
            df_mundo = client.query_df(q_mundo)
            df_mundo['entity_type'] = 'Mundo'
            df_mundo['entity_name'] = 'Mundo'
            
            # 2b. Región (Por año, tópico y región)
            q_region = f"""
            SELECT 
                year, topic, {region_mapping_sql} as entity_name,
                sum(doc_count) as doc_count, sum(fwci_sum) as fwci_sum, sum(percentile_sum) as percentile_sum,
                sum(top_10_sum) as top_10_sum, sum(top_1_sum) as top_1_sum,
                sum(gold_count) as gold_sum, sum(diamond_count) as diamond_sum, sum(green_count) as green_sum,
                sum(hybrid_count) as hybrid_sum, sum(bronze_count) as bronze_sum, sum(closed_count) as closed_sum,
                sum(lang_en) as lang_en_sum, sum(lang_es) as lang_es_sum, sum(lang_pt) as lang_pt_sum
            FROM summing_subfield_metrics WHERE subfield = '{subfield}' AND year >= 1900
            GROUP BY year, topic, entity_name
            """
            df_region = client.query_df(q_region)
            df_region['entity_type'] = 'Region'
            
            # 2c. País (Por año, tópico, región y país)
            q_pais = f"""
            SELECT 
                year, topic, country_code as entity_name,
                sum(doc_count) as doc_count, sum(fwci_sum) as fwci_sum, sum(percentile_sum) as percentile_sum,
                sum(top_10_sum) as top_10_sum, sum(top_1_sum) as top_1_sum,
                sum(gold_count) as gold_sum, sum(diamond_count) as diamond_sum, sum(green_count) as green_sum,
                sum(hybrid_count) as hybrid_sum, sum(bronze_count) as bronze_sum, sum(closed_count) as closed_sum,
                sum(lang_en) as lang_en_sum, sum(lang_es) as lang_es_sum, sum(lang_pt) as lang_pt_sum
            FROM summing_subfield_metrics WHERE subfield = '{subfield}' AND year >= 1900
            GROUP BY year, topic, entity_name
            """
            df_pais = client.query_df(q_pais)
            df_pais['entity_type'] = 'Country'
            
            # Unir todo
            df = pd.concat([df_mundo, df_region, df_pais], ignore_index=True)
            
        if df.empty:
            return False
            
        # 3. Post-procesamiento: Calcular ratios en Python
        df['fwci'] = np.where(df['doc_count'] > 0, df['fwci_sum'] / df['doc_count'], 0)
        df['percentile'] = np.where(df['doc_count'] > 0, (df['percentile_sum'] / df['doc_count']) * 100, 0)
        df['pct_top_10'] = np.where(df['doc_count'] > 0, (df['top_10_sum'] / df['doc_count']) * 100, 0)
        df['pct_top_1'] = np.where(df['doc_count'] > 0, (df['top_1_sum'] / df['doc_count']) * 100, 0)
        
        # OA & Lang
        for col in ['gold', 'diamond', 'green', 'hybrid', 'bronze', 'closed']:
            df[f'pct_oa_{col}'] = np.where(df['doc_count'] > 0, (df[f'{col}_sum'] / df['doc_count']) * 100, 0)
        
        for lang in ['en', 'es', 'pt']:
            df[f'pct_lang_{lang}'] = np.where(df['doc_count'] > 0, (df[f'lang_{lang}_sum'] / df['doc_count']) * 100, 0)

        # Guardar cache principal
        sub_clean = subfield.strip().replace(' ', '_').lower()
        cache_path = CACHE_TEMAS_DIR / f"{sub_clean}.parquet"
        df.to_parquet(cache_path, index=False)
        
        # 4. Query independiente para Top Journals
        journal_query = f"""
        SELECT 
            source_id as journal_id, 
            any(S.display_name) as Revista,
            sum(doc_count) as articulos
        FROM summing_subfield_metrics AS W
        LEFT JOIN (
            SELECT id, argMax(display_name, updated_date) as display_name 
            FROM sources GROUP BY id
        ) AS S ON W.source_id = S.id
        WHERE subfield = '{subfield}' AND year >= 2021
        GROUP BY journal_id
        ORDER BY articulos DESC
        LIMIT 100
        """
        df_journals = client.query_df(journal_query)
        df_journals = df_journals.rename(columns={'articulos': 'Artículos'})
        df_journals.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_journals.parquet", index=False)

        # 5. Colaboración Internacional
        try:
            collab_query = f"""
            SELECT 
                c1 as country_a, 
                c2 as country_b, 
                count() as count
            FROM (
                SELECT 
                    arrayJoin(arrayDistinct(all_country_codes)) as c1,
                    arrayDistinct(all_country_codes) as arr
                FROM works
                WHERE subfield = '{subfield}'
                  AND length(all_country_codes) > 1
                  AND publication_year >= 1900
            )
            ARRAY JOIN arr as c2
            WHERE c1 < c2
              AND c1 != '' AND c2 != ''
            GROUP BY c1, c2
            ORDER BY count DESC
            LIMIT 1000
            """
            df_collab = client.query_df(collab_query)
            df_collab.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_collab.parquet", index=False)
        except Exception as e:
            st.error(f"Error calculando colaboración: {e}")

        # 6. Analítica Institucional MULTI-SEGMENTO (Garantiza representatividad MX y Regional)
        try:
            # Query base para reutilizar
            q_base = f"""
            SELECT 
                W.year,
                W.institution_id,
                I.display_name as institution_name,
                I.country_code as country_code,
                {region_mapping_sql.replace('country_code', 'I.country_code')} as region,
                sum(W.doc_count) as doc_count,
                if(sum(W.doc_count) > 0, sum(W.fwci_sum) / sum(W.doc_count), 0) AS fwci,
                if(sum(W.doc_count) > 0, (sum(W.percentile_sum) / sum(W.doc_count)) * 100, 0) AS percentile,
                if(sum(W.doc_count) > 0, (sum(W.top_10_sum) / sum(W.doc_count)) * 100, 0) AS pct_top_10,
                if(sum(W.doc_count) > 0, (sum(W.top_1_sum) / sum(W.doc_count)) * 100, 0) AS pct_top_1,
                sum(W.citations_sum) as citations,
                sum(W.intl_collab_count) as intl_collab,
                sum(W.sdg_count) as sdg_docs,
                sum(W.award_count) as award_docs
            FROM summing_subfield_inst_metrics AS W
            LEFT JOIN institutions AS I ON W.institution_id = I.id
            WHERE W.subfield = '{subfield}' AND W.year >= 1900
            GROUP BY W.year, W.institution_id, institution_name, country_code, region
            """

            # Combinamos Global Top, MX Top y Líderes Regionales
            inst_query = f"""
            SELECT * FROM (
                -- 6a. Global Top 1000
                (SELECT *, 'Global' as segment FROM ({q_base}) ORDER BY doc_count DESC LIMIT 1000)
                UNION DISTINCT
                -- 6b. México Top 500 (Garantizado)
                (SELECT *, 'Mexico' as segment FROM ({q_base}) WHERE country_code = 'MX' ORDER BY doc_count DESC LIMIT 500)
                UNION DISTINCT
                -- 6c. Top 100 por Región (Garantiza representatividad LATAM, etc.)
                (
                    SELECT * EXCEPT(rn), 'Regional' as segment FROM (
                        SELECT *, row_number() OVER (PARTITION BY region ORDER BY doc_count DESC) as rn
                        FROM ({q_base})
                    ) WHERE rn <= 100
                )
            )
            """
            df_inst = client.query_df(inst_query)
            df_inst.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_inst.parquet", index=False)
        except Exception as e:
            st.error(f"Error calculando analítica institucional segmentada: {e}")

        # 7. Distribución de Tipos Documentales
        try:
            type_query = f"""
            SELECT 
                publication_year as year,
                country_code,
                type as doc_type,
                count() as count
            FROM works
            WHERE subfield = '{subfield}' AND publication_year >= 1900
            GROUP BY year, country_code, doc_type
            """
            df_types = client.query_df(type_query)
            df_types.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_types.parquet", index=False)
        except Exception as e:
            st.error(f"Error calculando distribución de tipos documentales: {e}")

        # 8. Distribución de Tipos de Instituciones
        try:
            inst_type_query = f"""
            SELECT 
                publication_year as year,
                country_code,
                arrayJoin(arrayDistinct(arrayFlatten(arrayMap(x -> arrayMap(i -> JSONExtractString(i, 'type'), x.institutions), 
                    JSONExtract(raw_data, 'authorships', 'Array(Tuple(institutions Array(String)))'))))) as inst_type,
                count() as count
            FROM works
            WHERE subfield = '{subfield}' AND publication_year >= 1900
              AND length(institution_ids) > 0
            GROUP BY year, country_code, inst_type
            """
            df_inst_types = client.query_df(inst_type_query)
            df_inst_types.to_parquet(CACHE_TEMAS_DIR / f"{sub_clean}_inst_types.parquet", index=False)
        except Exception as e:
            st.error(f"Error calculando distribución de tipos de instituciones: {e}")

        return True
    except Exception as e:
        st.error(f"Error en ClickHouse/Procesamiento: {e}")
        return False
