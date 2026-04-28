import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ClickHouse connection params
CH_HOST = os.environ.get('CH_HOST', 'localhost')
CH_PORT = int(os.environ.get('CH_PORT', 8123))
CH_USER = os.environ.get('CH_USER', 'default')
CH_PASSWORD = os.environ.get('CH_PASSWORD', '')
CH_DATABASE = os.environ.get('CH_DATABASE', 'rag')

def get_ch_client():
    """Retorna un cliente de ClickHouse con reintento sin SSL si falla el puerto 8124."""
    import clickhouse_connect
    is_secure = (CH_PORT == 8124)
    try:
        return clickhouse_connect.get_client(
            host=CH_HOST,
            port=CH_PORT,
            username=CH_USER,
            password=CH_PASSWORD,
            database=CH_DATABASE,
            secure=is_secure,
            verify=False
        )
    except Exception:
        if is_secure:
            return clickhouse_connect.get_client(
                host=CH_HOST,
                port=CH_PORT,
                username=CH_USER,
                password=CH_PASSWORD,
                database=CH_DATABASE,
                secure=False,
                verify=False
            )
        raise

def get_years_for_subfield(subfield_name):
    """Obtiene la distribución de años para un subcampo (agregada en ClickHouse, no descargando todo)."""
    client = get_ch_client()
    query = f"""
    SELECT publication_year, count() as n
    FROM works_flat
    WHERE subfield_name = '{subfield_name}'
    GROUP BY publication_year
    ORDER BY publication_year
    """
    df = client.query_df(query)
    if df.empty:
        return []
    # Expandir en Python: repetir cada año según su frecuencia
    # ClickHouse devuelve COUNT() como uint64; cast explícito a int64 para numpy
    import numpy as np
    return np.repeat(
        df['publication_year'].values.astype('int64'),
        df['n'].values.astype('int64')
    ).tolist()

def get_citation_pairs(subfield_name, year_start, year_end):
    """Obtiene los pares de citación usando INNER JOIN para evitar doble escaneo O(N²)."""
    client = get_ch_client()
    query = f"""
    SELECT W.id AS source_id, ref AS target_id
    FROM works_flat AS W
    ARRAY JOIN W.referenced_works AS ref
    INNER JOIN (
        SELECT id FROM works_flat
        WHERE subfield_name = '{subfield_name}'
    ) AS T ON ref = T.id
    WHERE W.subfield_name = '{subfield_name}'
      AND W.publication_year BETWEEN {year_start} AND {year_end}
    """
    return client.query_df(query)

def get_sandbox_data(subfield_name='Pulmonary and Respiratory Medicine', term='covid', limit=1000):
    """Extracción de datos optimizada usando la columna abstract materializada."""
    client = get_ch_client()
    query = f"""
    SELECT 
        id, 
        title, 
        abstract,
        publication_year, 
        referenced_works
    FROM works_flat
    WHERE subfield_name = '{subfield_name}'
      AND (hasToken(lower(title), '{term}') OR hasToken(lower(abstract), '{term}'))
      AND publication_year BETWEEN 2020 AND 2022
    ORDER BY id
    LIMIT {limit}
    """
    return client.query_df(query)

def get_work_metadata(work_ids):
    """Obtiene metadatos directamente de works_flat."""
    if not work_ids:
        return pd.DataFrame()
    client = get_ch_client()
    ids_str = ", ".join([f"'{wid}'" for wid in work_ids])
    query = f"""
    SELECT 
        id, title, publication_year, cited_by_count, fwci, abstract
    FROM works_flat
    WHERE id IN ({ids_str})
    """
    return client.query_df(query)


def get_bin_metadata(subfield_name: str, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Extrae todos los campos necesarios para el pipeline de una ventana temporal.
    Incluye referenced_works, author_ids, institution_ids y source_id para
    la construcción del grafo heterogéneo topológico.
    """
    client = get_ch_client()
    query = f"""
    SELECT
        id,
        title,
        abstract,
        publication_year,
        cited_by_count,
        fwci,
        referenced_works,
        institution_ids,
        source_id
    FROM works_flat
    WHERE subfield_name = '{subfield_name}'
      AND publication_year BETWEEN {year_start} AND {year_end}
    ORDER BY id
    """
    return client.query_df(query)


def get_citation_pairs_open(subfield_name: str, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Pares de citas para corpus ABIERTO: incluye TODAS las referencias citadas
    (no solo las que están dentro del mismo subcampo).
    Uso: construcción de la matriz A con referencias externas al subcampo
    para capturar frentes interdisciplinares.
    """
    client = get_ch_client()
    query = f"""
    SELECT W.id AS source_id, ref AS target_id
    FROM works_flat AS W
    ARRAY JOIN W.referenced_works AS ref
    WHERE W.subfield_name = '{subfield_name}'
      AND W.publication_year BETWEEN {year_start} AND {year_end}
      AND ref != ''
    """
    return client.query_df(query)
