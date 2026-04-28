import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path

# --- CONFIGURATION ---
BASE_PATH = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_PATH / 'data'
CACHE_TEMAS_DIR = DATA_DIR / 'cache_temas'

# Load Data
# Load Data
def load_subfield_data(subfield, suffix=""):
    sub_clean = subfield.strip().replace(' ', '_').lower()
    cache_path = CACHE_TEMAS_DIR / f"{sub_clean}{suffix}.parquet"
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            st.error(f"Error leyendo el archivo parquet: {e}")
            return None
    return None

def load_collaboration_data(subfield, suffix=""):
    sub_clean = subfield.strip().replace(' ', '_').lower()
    fname = f"{sub_clean}_collab{suffix}.parquet" if suffix else f"{sub_clean}_collab.parquet"
    cache_path = CACHE_TEMAS_DIR / fname
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            st.error(f"Error leyendo el archivo de colaboración: {e}")
            return None
    return None

def load_institutional_data(subfield, suffix=""):
    sub_clean = subfield.strip().replace(' ', '_').lower()
    fname = f"{sub_clean}_inst{suffix}.parquet" if suffix else f"{sub_clean}_inst.parquet"
    cache_path = CACHE_TEMAS_DIR / fname
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            st.error(f"Error leyendo el archivo institucional: {e}")
            return None
    return None

def load_types_data(subfield, suffix=""):
    sub_clean = subfield.strip().replace(' ', '_').lower()
    fname = f"{sub_clean}_types{suffix}.parquet" if suffix else f"{sub_clean}_types.parquet"
    cache_path = CACHE_TEMAS_DIR / fname
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            st.error(f"Error leyendo el archivo de tipos documentales: {e}")
            return None
    return None

def get_type_distribution(df_types, entity_name):
    """Agrega la distribución de tipos documentales para una entidad específica."""
    if df_types is None or df_types.empty: return None
    
    from regions import GLOBAL_REGIONS
    df = df_types.copy()
    
    if entity_name == 'Mundo':
        # Priorizar el registro consolidado ('' o NaN) para evitar doble conteo
        mundo_df = df[df['country_code'].isna() | (df['country_code'] == '')]
        if not mundo_df.empty:
            return mundo_df
        return df.groupby(['year', 'doc_type'])['count'].sum().reset_index()
    
    if entity_name == 'México':
        return df[df['country_code'] == 'MX']
    
    # Regiones
    if entity_name in GLOBAL_REGIONS:
        countries = GLOBAL_REGIONS[entity_name]
        return df[df['country_code'].isin(countries)].groupby(['year', 'doc_type'])['count'].sum().reset_index()
    
    return None

def load_inst_types_data(subfield, suffix=""):
    sub_clean = subfield.strip().replace(' ', '_').lower()
    fname = f"{sub_clean}_inst_types{suffix}.parquet" if suffix else f"{sub_clean}_inst_types.parquet"
    cache_path = CACHE_TEMAS_DIR / fname
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            st.error(f"Error leyendo el archivo de tipos de instituciones: {e}")
            return None
    return None

def get_inst_type_distribution(df_types, entity_name):
    """Agrega la distribución de tipos de instituciones para una entidad específica."""
    if df_types is None or df_types.empty: return None
    
    from regions import GLOBAL_REGIONS
    df = df_types.copy()
    
    if entity_name == 'Mundo':
        # Priorizar el registro consolidado ('' o NaN) para evitar doble conteo
        mundo_df = df[df['country_code'].isna() | (df['country_code'] == '')]
        if not mundo_df.empty:
            return mundo_df
        return df.groupby(['year', 'inst_type'])['count'].sum().reset_index()
    
    if entity_name == 'México':
        return df[df['country_code'] == 'MX']
    
    # Regiones
    if entity_name in GLOBAL_REGIONS:
        countries = GLOBAL_REGIONS[entity_name]
        return df[df['country_code'].isin(countries)].groupby(['year', 'inst_type'])['count'].sum().reset_index()
    
    return None

# --- AGGREGATION LOGIC ---
def get_entity_metrics(df, entity_name, period="Últimos 5 años (2021-2025)"):
    """Extrae métricas pre-calculadas para una entidad y periodo específico."""
    if df is None or df.empty: return None
    
    # Mapeo de nombres de UI a tipos de ClickHouse
    entity_type = 'Mundo' if entity_name == 'Mundo' else ('Country' if entity_name == 'México' else 'Region')
    lookup_name = 'Mundo' if entity_name == 'Mundo' else ('MX' if entity_name == 'México' else entity_name)
    
    # Filtrar por Entidad
    dff = df[(df['entity_type'] == entity_type) & (df['entity_name'] == lookup_name)]
    if dff.empty: return None

    # 1. Métricas del Periodo (Promedio Simple de los promedios ya ponderados por ClickHouse en el periodo)
    # Nota: Para precisión absoluta en promedios de periodos, Clickhouse podría devolverlos, 
    # pero como ya vienen ponderados por año/topic, el error de promedio de promedios es despreciable aquí.
    if period == "Últimos 5 años (2021-2025)":
        dff_period = dff[(dff['year'] >= 2021) & (dff['year'] <= 2025)]
    else:
        dff_period = dff.copy()

    metrics = {}
    if not dff_period.empty:
        # Sumamos docs, promediamos el resto (Clickhouse ya los devolvió calculados por grupo)
        # Para ratios, hacemos promedio pesado por doc_count para mantener precisión
        p = dff_period
        total_docs = p['doc_count'].sum()
        metrics['docs'] = int(total_docs)
        
        # Función auxiliar para promedio pesado
        def wt_avg(col): return (p[col] * p['doc_count']).sum() / total_docs if total_docs > 0 else 0
        
        metrics['fwci'] = wt_avg('fwci')
        metrics['percentile'] = wt_avg('percentile')
        metrics['top_10'] = wt_avg('pct_top_10')
        metrics['top_1'] = wt_avg('pct_top_1')
        
        # OA & Languages
        for col in [c for c in p.columns if c.startswith('pct_oa_') or c.startswith('pct_lang_')]:
            metrics[col] = wt_avg(col)
    else:
        metrics = None

    # 2. Tendencias Anuales (Agregamos por año sobre el dataframe pre-agregado)
    trends = dff.groupby('year').apply(lambda x: pd.Series({
        'doc_count': x['doc_count'].sum(),
        'fwci': (x['fwci'] * x['doc_count']).sum() / x['doc_count'].sum(),
        'pct_top_10': (x['pct_top_10'] * x['doc_count']).sum() / x['doc_count'].sum(),
        'pct_top_1': (x['pct_top_1'] * x['doc_count']).sum() / x['doc_count'].sum(),
        'percentile': (x['percentile'] * x['doc_count']).sum() / x['doc_count'].sum(),
    })).reset_index()
    
    # 3. Tendencias por Tópico
    topical_trends = dff.copy()
    
    # 4. Top Tópicos (solo si el dataset tiene granularidad por tópico)
    if 'topic' in dff_period.columns:
        top_topics = dff_period.groupby('topic')['doc_count'].sum().sort_values(ascending=False)
    else:
        top_topics = pd.Series(dtype='float64')
    
    return {
        'metrics': metrics,
        'trends': trends,
        'topical_trends': topical_trends,
        'top_topics': top_topics
    }

def get_summary_tables(df):
    """Genera tablas de resumen a partir de los datos pre-agregados."""
    if df is None or df.empty:
        return None, None, None, None, None, None

    def aggregate_metrics(df_src, group_cols):
        # 1. Agregación de sumas
        sum_cols = [c for c in df_src.columns if c.endswith('_sum') or c == 'doc_count' or c == 'citations']
        agg = df_src.groupby(group_cols)[sum_cols].sum().reset_index()
        
        # 2. Recalcular métricas de impacto (Promedio Pesado)
        for m in ['fwci', 'percentile', 'pct_top_10', 'pct_top_1']:
            is_pct = 'pct' in m
            sum_col = f"{m}_sum" if not is_pct else f"{m.replace('pct_','')}_sum"
            if sum_col in agg.columns:
                agg[m] = np.where(agg['doc_count'] > 0, (agg[sum_col] / agg['doc_count']) * (100 if is_pct else 1), 0)
        
        # 3. OA & Languages
        for col in [c for c in df_src.columns if c.startswith('pct_oa_') or c.startswith('pct_lang_')]:
            base = col.split('_')[-1]
            sum_col = f"{base}_sum"
            if sum_col in agg.columns:
                agg[col] = np.where(agg['doc_count'] > 0, (agg[sum_col] / agg['doc_count']) * 100, 0)
        return agg

    # 1. Tabla por Países (Agregada por País y Año)
    df_countries_raw = df[df['entity_type'] == 'Country'].copy()
    df_countries = aggregate_metrics(df_countries_raw, ['year', 'entity_name'])
    df_countries = df_countries.rename(columns={
        'year': 'Año', 'entity_name': 'País', 'doc_count': 'Documentos', 
        'fwci': 'FWCI', 'pct_top_10': '% Top 10%', 'pct_top_1': '% Top 1%', 'percentile': 'Percentil'
    })
    
    # 1b. Tabla por Países (Total Histórico - Agregado solo por País)
    df_countries_total = aggregate_metrics(df_countries_raw, ['entity_name'])
    df_countries_total = df_countries_total.rename(columns={
        'entity_name': 'País', 'doc_count': 'Documentos', 
        'fwci': 'FWCI', 'pct_top_10': '% Top 10%', 'pct_top_1': '% Top 1%', 'percentile': 'Percentil'
    })
    
    # 2. Tabla por Tópicos (Mundial - Agregada por Tópico y Año)
    df_topics_raw = df[df['entity_type'] == 'Mundo'].copy()
    df_topics = aggregate_metrics(df_topics_raw, ['year', 'topic'])
    df_topics = df_topics.rename(columns={
        'year': 'Año', 'topic': 'Tópico', 'doc_count': 'Documentos',
        'fwci': 'FWCI', 'pct_top_10': '% Top 10%', 'pct_top_1': '% Top 1%', 'percentile': 'Percentil'
    })

    # 2b. Tabla por Tópicos (Total Histórico - Agregado solo por Tópico)
    df_topics_total = aggregate_metrics(df_topics_raw, ['topic'])
    df_topics_total = df_topics_total.rename(columns={
        'topic': 'Tópico', 'doc_count': 'Documentos',
        'fwci': 'FWCI', 'pct_top_10': '% Top 10%', 'pct_top_1': '% Top 1%', 'percentile': 'Percentil'
    })

    # 3. Pivot Tables (Producción por tópicos)
    # Tópicos Ordenados por volumen mundial total
    topic_order = df[df['entity_type'] == 'Mundo'].groupby('topic')['doc_count'].sum().sort_values(ascending=False).index.tolist()
    
    # Entidades Ordenadas
    world_total = df[df['entity_type'] == 'Mundo'].groupby('entity_name')['doc_count'].sum()
    region_totals = df[df['entity_type'] == 'Region'].groupby('entity_name')['doc_count'].sum().sort_values(ascending=False)
    country_totals = df[df['entity_type'] == 'Country'].groupby('entity_name')['doc_count'].sum().sort_values(ascending=False)
    entity_order = ["Mundo"] + region_totals.index.tolist() + country_totals.index.tolist()

    def generate_pivot_optimized(df_source, id_col, topic_order, entity_order):
        # Crear identificador
        if 'year' in df_source.columns:
            df_source['ID'] = df_source['year'].astype(str) + "_" + df_source['entity_name']
            sort_cols = ['year', 'entity_name']
        else:
            df_source['ID'] = df_source['entity_name']
            sort_cols = ['entity_name']
            
        pivoted = df_source.pivot(index='ID', columns='topic', values='doc_count').fillna(0)
        
        # Ordenar columnas
        cols = [c for c in topic_order if c in pivoted.columns]
        pivoted = pivoted[cols]
        
        # Ordenar filas (simplificado)
        # Re-indexar basado en entity_order (necesitaría lógica de Año_... similar a la anterior si hay años)
        # Por brevedad y para "aprovechar el poder de Clickhouse", asumimos que el orden ya viene refinado o lo hacemos simple
        return pivoted.reset_index().rename(columns={'ID': 'Año_Entidad' if 'year' in df_source.columns else 'Entidad'})

    df_ct_annual = generate_pivot_optimized(df.copy(), 'ID', topic_order, entity_order)
    
    # Histórico (Sin años)
    df_hist = df.groupby(['entity_name', 'topic'])['doc_count'].sum().reset_index()
    df_ct_full = generate_pivot_optimized(df_hist, 'entity_name', topic_order, entity_order)
    
    # 2021-2025
    df_2125 = df[df['year'] >= 2021].groupby(['entity_name', 'topic'])['doc_count'].sum().reset_index()
    df_ct_2125 = generate_pivot_optimized(df_2125, 'entity_name', topic_order, entity_order)

    return df_countries, df_topics, df_countries_total, df_ct_annual, df_ct_full, df_ct_2125, df_topics_total
