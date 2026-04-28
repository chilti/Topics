"""
fronts/parallel.py
Ejecución paralela del pipeline de frentes de investigación.

Estrategia:
  1. Pre-embeber todos los papers (GPU, secuencial) → embeddings_cache ClickHouse
  2. Procesar bins en paralelo (ProcessPoolExecutor, N_BIN_WORKERS workers)
     - Dentro de cada bin: structural + topológico en ThreadPool paralelo
  3. AMI + Tracking secuencial (requiere todos los bins)

Diagrama de dependencias por bin:
  ClickHouse metadata ──┬──► Leiden (structural)   ─┐
                        │                            ├──► AMI ──► Tracking
                        └──► FastRP (topological)   ─┤
  embeddings_cache ────────► UMAP → HDBSCAN         ─┘
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Worker de nivel módulo (picklable → compatible con ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _bin_worker(kwargs: dict) -> dict:
    """
    Procesa un bin completo de forma independiente.
    Función de nivel módulo → picklable por ProcessPoolExecutor.

    Returns: dict con bin_id y ruta al parquet de resultados del bin.
    """
    # Importaciones dentro del worker para evitar conflictos entre procesos
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import adjusted_mutual_info_score
    from concurrent.futures import ThreadPoolExecutor

    # Añadir root al path del subproceso
    root = kwargs['root_path']
    if root not in sys.path:
        sys.path.insert(0, root)

    from fronts.config import (
        SALTON_THRESHOLD, LEIDEN_RESOLUTION, OPEN_CORPUS,
        UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_METRIC, UMAP_MIN_DIST,
        UMAP_N_JOBS,
        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC,
        FASTRP_DIMENSION, FASTRP_ITERATIONS,
        WITHIN_BIN_PARALLEL,
        get_window_cache_dir
    )
    from fronts.clickhouse_queries import get_citation_pairs, get_bin_metadata
    from fronts.structural.citation_network import (
        build_citation_matrix, normalize_salton, apply_salton_threshold
    )
    from fronts.structural.leiden_detector import run_leiden
    from fronts.semantic.dimensionality import reduce_dimensions
    from fronts.semantic.hdbscan_detector import run_hdbscan
    from fronts.topological.fastrp_detector import (
        build_heterogeneous_graph, run_fastrp_igraph_sparse
    )
    from fronts.embeddings.cache_manager import get_embeddings_for_window

    subfield  = kwargs['subfield_name']
    sub_clean = kwargs['sub_clean']
    bin_id    = kwargs['bin_id']
    y_start   = kwargs['y_start']
    y_end     = kwargs['y_end']
    force     = kwargs.get('force_levels', set())  # set de niveles a forzar

    win_dir = get_window_cache_dir(sub_clean, bin_id)
    pid = os.getpid()
    prefix = f"[PID {pid}][Bin {bin_id:03d} {y_start}-{y_end}]"

    def _load(path):
        return pd.read_parquet(path) if Path(path).exists() else None

    def _save(df, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def _load_npy(path):
        return np.load(path, allow_pickle=True) if Path(path).exists() else None

    def _save_npy(arr, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    # ── Metadata + citas ─────────────────────────────────────────────────────
    meta_path = win_dir / "metadata.parquet"
    cit_path  = win_dir / "citations.parquet"

    df_meta = _load(meta_path) if 'citations' not in force else None
    df_cit  = _load(cit_path)  if 'citations' not in force else None

    if df_meta is None:
        print(f"{prefix} Extrayendo metadata...")
        df_meta = get_bin_metadata(subfield, y_start, y_end)
        _save(df_meta, meta_path)
    if df_cit is None:
        df_cit = get_citation_pairs(subfield, y_start, y_end)
        _save(df_cit, cit_path)

    if df_meta is None or df_meta.empty:
        print(f"{prefix} Sin datos, saltando.")
        return {'bin_id': bin_id, 'result_path': None}

    # ── Función: Leiden (structural) ─────────────────────────────────────────
    def compute_leiden():
        leiden_path = win_dir / "structural" / "leiden.parquet"
        if leiden_path.exists() and 'structural' not in force:
            return pd.read_parquet(leiden_path)
        print(f"{prefix} Leiden...")
        if 'referenced_works' in df_meta.columns:
            C_BC, work_to_idx = build_citation_matrix(df_meta, open_corpus=OPEN_CORPUS)
        else:
            C_BC, work_to_idx = None, {}
        if C_BC is not None:
            S = normalize_salton(C_BC)
            S = apply_salton_threshold(S, SALTON_THRESHOLD)
            clusters = run_leiden(S, resolution=LEIDEN_RESOLUTION)
            idx_to_id = {v: k for k, v in work_to_idx.items()}
            df_l = pd.DataFrame({
                'id': [idx_to_id[i] for i in range(len(clusters))],
                'cluster_leiden': clusters
            })
        else:
            df_l = pd.DataFrame({'id': df_meta['id'].tolist(),
                                  'cluster_leiden': [-1] * len(df_meta)})
        df_l.to_parquet(leiden_path, index=False)
        leiden_path.parent.mkdir(parents=True, exist_ok=True)
        return df_l

    # ── Función: FastRP (topological) ────────────────────────────────────────
    def compute_fastrp():
        fastrp_path = win_dir / "topological" / "fastrp.npy"
        hdb_top_path = win_dir / "topological" / "hdbscan.parquet"
        if hdb_top_path.exists() and 'topological' not in force:
            return pd.read_parquet(hdb_top_path)
        vecs = _load_npy(fastrp_path) if ('topological' not in force) else None
        if vecs is None:
            print(f"{prefix} FastRP (igraph)...")
            g, work_ids = build_heterogeneous_graph(df_cit, df_meta)
            vecs = run_fastrp_igraph_sparse(g, len(work_ids), FASTRP_DIMENSION, FASTRP_ITERATIONS)
            _save_npy(vecs, fastrp_path)
        print(f"{prefix} HDBSCAN topológico...")
        top_labels, _ = run_hdbscan(vecs, HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC)
        g_ids = df_meta['id'].tolist()[:vecs.shape[0]]
        df_t = pd.DataFrame({'id': g_ids, 'cluster_topological': top_labels})
        hdb_top_path.parent.mkdir(parents=True, exist_ok=True)
        df_t.to_parquet(hdb_top_path, index=False)
        return df_t

    # ── Función: UMAP + HDBSCAN (semantic) ───────────────────────────────────
    def compute_semantic():
        umap_path    = win_dir / "semantic" / "umap30d.npy"
        ids_path     = win_dir / "semantic" / "ids.npy"
        hdb_sem_path = win_dir / "semantic" / "hdbscan.parquet"

        if hdb_sem_path.exists() and 'semantic' not in force:
            return pd.read_parquet(hdb_sem_path)

        projections = _load_npy(umap_path) if ('umap' not in force) else None
        if projections is None:
            print(f"{prefix} Recuperando embeddings SPECTER2...")
            df_emb = get_embeddings_for_window(subfield, y_start, y_end, 'embedding_specter2')
            if df_emb.empty:
                return None
            paper_ids = df_emb['id'].tolist()
            embeddings = np.vstack(df_emb['embedding'].tolist()).astype(np.float32)

            print(f"{prefix} UMAP {embeddings.shape[1]}d → {UMAP_N_COMPONENTS}d...")
            umap_kwargs = dict(
                n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                metric=UMAP_METRIC, min_dist=UMAP_MIN_DIST, low_memory=False
            )
            if UMAP_N_JOBS is not None:
                umap_kwargs['n_jobs'] = UMAP_N_JOBS
                # Sin random_state para permitir paralelismo dentro del worker
            else:
                umap_kwargs['random_state'] = 42
            import umap as umap_lib
            projections = umap_lib.UMAP(**umap_kwargs).fit_transform(embeddings)
            umap_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(umap_path, projections)
            np.save(ids_path, np.array(paper_ids, dtype=object))
        else:
            paper_ids = _load_npy(ids_path).tolist()

        print(f"{prefix} HDBSCAN semántico...")
        sem_labels, _ = run_hdbscan(projections, HDBSCAN_MIN_CLUSTER_SIZE,
                                    HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC)
        df_s = pd.DataFrame({'id': paper_ids, 'cluster_semantic': sem_labels})
        hdb_sem_path.parent.mkdir(parents=True, exist_ok=True)
        df_s.to_parquet(hdb_sem_path, index=False)
        return df_s

    # ── Ejecutar structural + topológico en paralelo (mismo proceso) ──────────
    if WITHIN_BIN_PARALLEL:
        with ThreadPoolExecutor(max_workers=2) as tex:
            f_leiden = tex.submit(compute_leiden)
            f_fastrp = tex.submit(compute_fastrp)
            df_leiden   = f_leiden.result()
            df_hdb_top  = f_fastrp.result()
    else:
        df_leiden  = compute_leiden()
        df_hdb_top = compute_fastrp()

    df_hdb_sem = compute_semantic()

    # ── AMI del bin ───────────────────────────────────────────────────────────
    ami_path = win_dir / "ami.parquet"
    ami_records = {'bin_id': bin_id, 'y_start': y_start, 'y_end': y_end}
    df_merged = df_meta[['id']].copy()
    for df_cl, col in [(df_leiden, 'cluster_leiden'),
                        (df_hdb_sem, 'cluster_semantic') if df_hdb_sem is not None else (None, None),
                        (df_hdb_top, 'cluster_topological')]:
        if df_cl is not None and col and col in df_cl.columns:
            df_merged = df_merged.merge(df_cl[['id', col]], on='id', how='left')

    for c1, c2 in [('cluster_leiden', 'cluster_semantic'),
                    ('cluster_leiden', 'cluster_topological'),
                    ('cluster_semantic', 'cluster_topological')]:
        if c1 in df_merged.columns and c2 in df_merged.columns:
            valid = df_merged[[c1, c2]].dropna()
            valid = valid[(valid[c1] != -1) & (valid[c2] != -1)]
            if len(valid) > 10:
                ami_records[f'ami_{c1}_vs_{c2}'] = adjusted_mutual_info_score(valid[c1], valid[c2])

    pd.DataFrame([ami_records]).to_parquet(ami_path, index=False)

    # ── Consolidar resultado del bin ──────────────────────────────────────────
    df_result = df_meta[['id', 'publication_year']].copy()
    df_result['bin_id'] = bin_id
    df_result['y_start'] = y_start
    df_result['y_end'] = y_end

    for df_cl, col in [(df_leiden, 'cluster_leiden'),
                        (df_hdb_sem, 'cluster_semantic') if df_hdb_sem is not None else (None, None),
                        (df_hdb_top, 'cluster_topological')]:
        if df_cl is not None and col and col in df_cl.columns:
            df_result = df_result.merge(df_cl[['id', col]], on='id', how='left')

    result_path = win_dir / "result.parquet"
    df_result.to_parquet(result_path, index=False)
    print(f"{prefix} DONE — {len(df_result):,} papers.")
    return {'bin_id': bin_id, 'result_path': str(result_path)}


# ---------------------------------------------------------------------------
# Orquestador paralelo
# ---------------------------------------------------------------------------

def run_bins_parallel(
    subfield_name: str,
    df_windows: pd.DataFrame,
    force_from: Optional[str] = None,
    n_workers: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Procesa todos los bins del pipeline en paralelo usando ProcessPoolExecutor.

    Args:
        subfield_name: Nombre del subcampo.
        df_windows: DataFrame con columnas [bin_id, y_start, y_end, mode].
        force_from: Nivel desde el que forzar recálculo (propaga a workers).
        n_workers: Número de procesos paralelos. None = N_BIN_WORKERS de config.
        verbose: Imprimir progreso.

    Returns:
        DataFrame consolidado con todos los papers y sus asignaciones de cluster.
    """
    from fronts.config import N_BIN_WORKERS, PIPELINE_LEVELS

    if n_workers is None:
        n_workers = N_BIN_WORKERS

    sub_clean = subfield_name.strip().lower().replace(' ', '_')
    root_path = str(_ROOT)
    log = print if verbose else lambda *a, **k: None

    # Determinar qué niveles forzar
    force_levels = set()
    if force_from and force_from in PIPELINE_LEVELS:
        idx = PIPELINE_LEVELS.index(force_from)
        force_levels = set(PIPELINE_LEVELS[idx:])

    # Preparar kwargs para cada bin (deben ser picklables)
    jobs = [
        {
            'subfield_name': subfield_name,
            'sub_clean': sub_clean,
            'bin_id': int(row['bin_id']),
            'y_start': int(row['y_start']),
            'y_end': int(row['y_end']),
            'force_levels': force_levels,
            'root_path': root_path,
        }
        for _, row in df_windows.iterrows()
    ]

    n_bins = len(jobs)
    effective_workers = min(n_workers, n_bins)
    log(f"\n🚀 Procesando {n_bins} bins con {effective_workers} workers paralelos...")

    result_paths = [None] * n_bins

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        future_to_bin = {executor.submit(_bin_worker, job): job['bin_id'] for job in jobs}
        completed = 0
        for future in as_completed(future_to_bin):
            bin_id = future_to_bin[future]
            try:
                result = future.result()
                result_paths[bin_id] = result.get('result_path')
                completed += 1
                log(f"  [{completed}/{n_bins}] Bin {bin_id:03d} completado.")
            except Exception as e:
                log(f"  ❌ Bin {bin_id:03d} falló: {e}")
                import traceback
                traceback.print_exc()

    # Consolidar todos los resultados
    dfs = []
    for path in result_paths:
        if path and Path(path).exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    log(f"\n✅ Paralelo completo: {len(df_all):,} papers en {len(dfs)} bins.")
    return df_all


def prefetch_embeddings(
    subfield_name: str,
    df_windows: pd.DataFrame,
    model_col: str = 'embedding_specter2',
    batch_size: int = 256,
    verbose: bool = True
):
    """
    Fase 1 (secuencial, GPU): genera y cachea embeddings SPECTER2 para
    todos los papers de todas las ventanas antes de iniciar el procesamiento paralelo.

    Llamar esto ANTES de run_bins_parallel() para maximizar la utilización de GPU.
    """
    from fronts.embeddings.cache_manager import get_missing_ids, insert_embeddings
    from fronts.semantic.embeddings import generate_specter_embeddings, prepare_text_for_specter
    from fronts.clickhouse_queries import get_bin_metadata

    log = print if verbose else lambda *a, **k: None
    log(f"\n⚡ Fase 1: Pre-fetch de embeddings para {len(df_windows)} ventanas...")

    all_missing_ids = set()
    for _, row in df_windows.iterrows():
        missing = get_missing_ids(subfield_name, int(row['y_start']), int(row['y_end']), model_col)
        all_missing_ids.update(missing)

    if not all_missing_ids:
        log("   ✅ Todos los embeddings ya están en cache.")
        return

    log(f"   Generando {len(all_missing_ids):,} embeddings nuevos (SPECTER2)...")
    # Buscar metadata de los papers que faltan
    year_min = int(df_windows['y_start'].min())
    year_max = int(df_windows['y_end'].max())
    df_meta  = get_bin_metadata(subfield_name, year_min, year_max)
    df_new   = df_meta[df_meta['id'].isin(all_missing_ids)][['id', 'title', 'abstract', 'publication_year']]

    if df_new.empty:
        log("   ⚠️  No se encontraron papers para embeber.")
        return

    texts = prepare_text_for_specter(df_new)
    embs  = generate_specter_embeddings(texts, batch_size=batch_size)
    insert_embeddings(
        df_new['id'].tolist(), embs, subfield_name,
        df_new['publication_year'].tolist(), model_col
    )
    log(f"   ✅ {len(df_new):,} embeddings generados y cacheados.")
