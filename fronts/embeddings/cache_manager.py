"""
embeddings/cache_manager.py
Gestiona el ciclo de vida de los embeddings en ClickHouse (tabla embeddings_cache).

Diseño: wide-format (una columna por modelo), ReplacingMergeTree.
- ADD COLUMN es de metadata (ms), no requiere mutaciones.
- Solo INSERTs — nunca UPDATE/mutation sobre works_flat.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fronts.clickhouse_queries import get_ch_client


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL_EMBEDDINGS_CACHE = """
CREATE TABLE IF NOT EXISTS embeddings_cache (
    id                   String          COMMENT 'OpenAlex Work ID',
    subfield_name        LowCardinality(String),
    publication_year     UInt16,

    -- Modelos semánticos (768d)
    embedding_specter2   Array(Float32)  DEFAULT [],
    embedding_scilbert   Array(Float32)  DEFAULT [],

    -- Modelos topológicos (128d)
    embedding_fastrp_cit Array(Float32)  DEFAULT [],   -- solo red de citas
    embedding_fastrp_het Array(Float32)  DEFAULT [],   -- red heterogénea (citas+autores+inst+fuente)

    -- Proyecciones reducidas
    embedding_umap_30d   Array(Float32)  DEFAULT [],   -- 30d derivado de specter2

    -- Auditoría por modelo
    specter2_at          Nullable(DateTime),
    scilbert_at          Nullable(DateTime),
    fastrp_cit_at        Nullable(DateTime),
    fastrp_het_at        Nullable(DateTime),
    umap_30d_at          Nullable(DateTime),

    updated_at           DateTime        DEFAULT now()

) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (subfield_name, publication_year, id)
COMMENT 'Cache de embeddings ML por paper. Una columna por modelo. Nunca modifica works_flat.';
"""


def ensure_table_exists():
    """Crea la tabla embeddings_cache si no existe."""
    client = get_ch_client()
    client.command(DDL_EMBEDDINGS_CACHE)
    print("✅ embeddings_cache lista.")


def add_model_column(model_col: str, dim: int, comment: str = ""):
    """
    Agrega una columna de embedding para un nuevo modelo.
    ALTER TABLE ADD COLUMN es de metadata — no reescribe datos.

    Args:
        model_col: nombre de la columna, ej. 'embedding_bge_m3'
        dim: dimensión esperada (solo informativo, en el comentario)
        comment: descripción del modelo
    """
    client = get_ch_client()
    col_comment = comment or f"{dim}d"
    audit_col = model_col.replace("embedding_", "") + "_at"

    client.command(f"""
        ALTER TABLE embeddings_cache
        ADD COLUMN IF NOT EXISTS {model_col} Array(Float32) DEFAULT []
        COMMENT '{col_comment}'
    """)
    client.command(f"""
        ALTER TABLE embeddings_cache
        ADD COLUMN IF NOT EXISTS {audit_col} Nullable(DateTime)
    """)
    print(f"✅ Columna '{model_col}' ({dim}d) agregada a embeddings_cache.")


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def get_missing_ids(
    subfield_name: str,
    year_start: int,
    year_end: int,
    model_col: str = "embedding_specter2"
) -> List[str]:
    """
    Retorna IDs de papers que NO tienen embedding para el modelo dado.
    Usa un LEFT JOIN para identificar el gap entre works_flat y embeddings_cache.
    """
    client = get_ch_client()
    query = f"""
    SELECT w.id
    FROM works_flat AS w
    LEFT JOIN (
        SELECT id
        FROM embeddings_cache
        WHERE subfield_name = '{subfield_name}'
          AND length({model_col}) > 0
    ) AS e ON w.id = e.id
    WHERE w.subfield_name = '{subfield_name}'
      AND w.publication_year BETWEEN {year_start} AND {year_end}
      AND e.id IS NULL
    """
    df = client.query_df(query)
    return df['id'].tolist() if not df.empty else []


def get_embeddings_for_window(
    subfield_name: str,
    year_start: int,
    year_end: int,
    model_col: str = "embedding_specter2"
) -> pd.DataFrame:
    """
    Recupera embeddings de una ventana temporal para el modelo dado.
    Solo lee la columna del modelo solicitado (columnar I/O mínimo).

    Returns:
        DataFrame con columnas ['id', 'embedding']
    """
    client = get_ch_client()
    query = f"""
    SELECT id, {model_col} AS embedding
    FROM embeddings_cache
    WHERE subfield_name = '{subfield_name}'
      AND publication_year BETWEEN {year_start} AND {year_end}
      AND length({model_col}) > 0
    """
    return client.query_df(query)


def get_coverage_report(subfield_name: str) -> pd.DataFrame:
    """Reporte de cobertura: cuántos papers tienen cada modelo."""
    client = get_ch_client()
    query = f"""
    SELECT
        countIf(length(embedding_specter2) > 0)   AS n_specter2,
        countIf(length(embedding_scilbert) > 0)   AS n_scilbert,
        countIf(length(embedding_fastrp_cit) > 0) AS n_fastrp_cit,
        countIf(length(embedding_fastrp_het) > 0) AS n_fastrp_het,
        countIf(length(embedding_umap_30d) > 0)   AS n_umap_30d,
        count()                                    AS n_total,
        max(updated_at)                            AS last_update
    FROM embeddings_cache
    WHERE subfield_name = '{subfield_name}'
    """
    return client.query_df(query)


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def insert_embeddings(
    ids: List[str],
    embeddings: np.ndarray,
    subfield_name: str,
    publication_years: List[int],
    model_col: str = "embedding_specter2",
    batch_size: int = 5000
):
    """
    Inserta embeddings en la tabla. Solo INSERTs, nunca mutations.
    ReplacingMergeTree se encarga de deduplicar por (subfield_name, publication_year, id).

    Args:
        ids: lista de OpenAlex IDs
        embeddings: array (n, dim) de float32
        subfield_name: nombre del subcampo
        publication_years: año de publicación de cada paper
        model_col: columna destino, ej. 'embedding_specter2'
        batch_size: papers por INSERT (evitar timeouts)
    """
    from datetime import datetime
    client = get_ch_client()
    audit_col = model_col.replace("embedding_", "") + "_at"
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    n = len(ids)
    inserted = 0
    for start in range(0, n, batch_size):
        batch_ids = ids[start:start + batch_size]
        batch_emb = embeddings[start:start + batch_size]
        batch_years = publication_years[start:start + batch_size]

        rows = [
            {
                'id': bid,
                'subfield_name': subfield_name,
                'publication_year': yr,
                model_col: emb.tolist(),
                audit_col: now_str,
                'updated_at': now_str,
            }
            for bid, yr, emb in zip(batch_ids, batch_years, batch_emb)
        ]
        client.insert('embeddings_cache', rows,
                      column_names=['id', 'subfield_name', 'publication_year',
                                    model_col, audit_col, 'updated_at'])
        inserted += len(rows)
        print(f"   Insertados {inserted}/{n} embeddings ({model_col})...")

    print(f"✅ {n} embeddings insertados en embeddings_cache[{model_col}].")
