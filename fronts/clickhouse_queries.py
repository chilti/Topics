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
    """Retorna un cliente de ClickHouse configurado."""
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE
    )

def get_years_for_subfield(subfield_name):
    """Obtiene la distribución de años para un subcampo para calcular los bins."""
    client = get_ch_client()
    query = f"""
    SELECT publication_year
    FROM works
    WHERE subfield = '{subfield_name}'
      AND publication_year IS NOT NULL
    ORDER BY publication_year
    """
    df = client.query_df(query)
    return df['publication_year'].tolist()

def get_citation_pairs(subfield_name, year_start, year_end):
    """Obtiene los pares de citación para construir la matriz de acoplamiento."""
    client = get_ch_client()
    # Usamos JSONExtract para referenced_works ya que no está materializado
    query = f"""
    SELECT
        id AS source_id,
        arrayJoin(JSONExtractArrayRaw(raw_data, 'referenced_works')) AS target_id
    FROM works
    WHERE subfield = '{subfield_name}'
      AND publication_year BETWEEN {year_start} AND {year_end}
      AND target_id IN (
          SELECT id FROM works
          WHERE subfield = '{subfield_name}'
      )
    """
    return client.query_df(query)

def get_sandbox_data(subfield_name='Pulmonary and Respiratory Medicine', term='covid', limit=1000):
    """Extracción de datos para el esquema de pruebas (Sandbox)."""
    client = get_ch_client()
    # Usamos JSONExtract para abstract y referenced_works
    query = f"""
    SELECT 
        id, 
        title, 
        JSONExtractString(raw_data, 'abstract_inverted_index') AS abstract_raw,
        publication_year, 
        JSONExtractArrayRaw(raw_data, 'referenced_works') AS referenced_works
    FROM works
    WHERE subfield = '{subfield_name}'
      AND (hasToken(lower(title), '{term}') OR hasToken(lower(raw_data), '{term}'))
      AND publication_year BETWEEN 2020 AND 2022
    ORDER BY id
    LIMIT {limit}
    """
    return client.query_df(query)

def get_work_metadata(work_ids):
    """Obtiene metadatos (FWCI, Citaciones, etc.) para una lista de IDs."""
    client = get_ch_client()
    ids_str = ", ".join([f"'{wid}'" for wid in work_ids])
    query = f"""
    SELECT 
        id, title, publication_year, cited_by_count, fwci
    FROM works
    WHERE id IN ({ids_str})
    """
    return client.query_df(query)
