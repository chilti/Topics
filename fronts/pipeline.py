"""
pipeline.py — Orquestador principal del pipeline de frentes de investigación.

Sistema de checkpoints por nivel:
  Nivel 1: windows, citations
  Nivel 2: structural, umap, semantic, topological
  Nivel 3: ami, tracking
  Nivel 4: labeling

Uso:
    # Corrida completa
    result = run_fronts_analysis('Pulmonary and Respiratory Medicine')

    # Re-correr solo desde HDBSCAN semántico (conserva UMAP cacheado)
    result = run_fronts_analysis('Pulmonary and Respiratory Medicine', force_from='semantic')

    # Re-correr todo desde cero
    result = run_fronts_analysis('...', force_from='windows')
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_mutual_info_score

# Asegurar que el directorio raíz está en el path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fronts.config import (
    K_BINS, WINDOW_YEARS, WINDOW_STEP, RECENT_FROM,
    SALTON_THRESHOLD, LEIDEN_RESOLUTION, OPEN_CORPUS,
    SPECTER_MODEL, SPECTER_BATCH_SIZE,
    UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_METRIC, UMAP_MIN_DIST,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_METRIC,
    FASTRP_DIMENSION, FASTRP_ITERATIONS,
    JACCARD_THRESHOLD, JACCARD_MIN,
    PIPELINE_LEVELS,
    get_window_cache_dir, get_subfield_cache_dir,
    N_BIN_WORKERS
)
from fronts.clickhouse_queries import get_years_for_subfield, get_citation_pairs
from fronts.segmentation.temporal_bins import (
    compute_temporal_bins, compute_sliding_windows, assign_bins_vectorized
)
from fronts.structural.citation_network import (
    build_citation_matrix, normalize_salton, apply_salton_threshold
)
from fronts.structural.leiden_detector import run_leiden
from fronts.semantic.embeddings import generate_specter_embeddings, prepare_text_for_specter
from fronts.semantic.dimensionality import reduce_dimensions
from fronts.semantic.hdbscan_detector import run_hdbscan
from fronts.topological.fastrp_detector import (
    build_heterogeneous_graph, run_fastrp_igraph_sparse
)
from fronts.longitudinal.cluster_tracker import track_clusters, calculate_jaccard


# ---------------------------------------------------------------------------
# Utilidades de checkpoint
# ---------------------------------------------------------------------------

def _should_force(force_from: str | None, level: str) -> bool:
    """Determina si un nivel debe re-calcularse según force_from."""
    if force_from is None:
        return False
    if force_from not in PIPELINE_LEVELS or level not in PIPELINE_LEVELS:
        return False
    return PIPELINE_LEVELS.index(level) >= PIPELINE_LEVELS.index(force_from)


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _load_npy(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path)
    return None


def _save_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_fronts_analysis(
    subfield_name: str,
    mode: str = 'vigintiles',         # 'vigintiles' | 'sliding' | 'both'
    force_from: str | None = None,    # Nivel desde el que forzar recálculo
    n_workers: int | None = None,     # None = N_BIN_WORKERS de config; 1 = secuencial
    prefetch: bool = True,            # Pre-embeber todos los papers antes del paralelo
    verbose: bool = True
) -> pd.DataFrame:
    """
    Orquestador principal. Ejecuta el triple pipeline por ventana temporal
    con sistema de checkpoints y paralelismo configurable.

    Args:
        subfield_name: Nombre del subcampo OpenAlex.
        mode: 'vigintiles' | 'sliding' | 'both'.
        force_from: Nivel a partir del cual forzar recálculo. None = usar cache.
        n_workers: Procesos paralelos para el procesamiento de bins.
                   None = usa N_BIN_WORKERS de config (N_PHYSICAL_CORES // 2).
                   1    = secuencial (debug, menos RAM).
                   -1   = todos los cores disponibles.
        prefetch: Si True, genera embeddings SPECTER2 en una pasada GPU
                  antes de iniciar el procesamiento paralelo.
        verbose: Imprimir progreso.

    Returns:
        DataFrame con clusters por paper, bin y método.
    """
    import os
    if n_workers is None:
        n_workers = N_BIN_WORKERS
    elif n_workers == -1:
        n_workers = os.cpu_count() or 4

    sub_clean = subfield_name.strip().lower().replace(' ', '_')
    sub_dir = get_subfield_cache_dir(sub_clean)
    log = print if verbose else lambda *a, **k: None

    # =========================================================================
    # NIVEL 1A: Particionamiento temporal
    # =========================================================================
    windows_path = sub_dir / "windows.parquet"
    df_windows = None if _should_force(force_from, 'windows') else _load_parquet(windows_path)

    if df_windows is None:
        log(f"\n📅 Nivel 1a: Calculando bins temporales para '{subfield_name}'...")
        years = get_years_for_subfield(subfield_name)
        if not years:
            log("❌ No hay datos para este subcampo.")
            return pd.DataFrame()

        bins_vigintiles = compute_temporal_bins(years, k=K_BINS) if mode in ('vigintiles', 'both') else []
        year_min, year_max = min(years), max(years)
        bins_sliding = compute_sliding_windows(
            max(RECENT_FROM, year_min), year_max, WINDOW_YEARS, WINDOW_STEP
        ) if mode in ('sliding', 'both') else []

        # Combinar y deduplicar bins
        all_bins = []
        for b in bins_vigintiles:
            all_bins.append({'bin_id': len(all_bins), 'y_start': b[0], 'y_end': b[1], 'mode': 'vigintile'})
        for b in bins_sliding:
            all_bins.append({'bin_id': len(all_bins), 'y_start': b[0], 'y_end': b[1], 'mode': 'sliding'})

        df_windows = pd.DataFrame(all_bins)
        _save_parquet(df_windows, windows_path)
        log(f"   ✅ {len(df_windows)} bins ({len(bins_vigintiles)} vigintiles + {len(bins_sliding)} sliding)")
    else:
        log(f"📦 Bins desde cache: {len(df_windows)} ventanas.")

    # =========================================================================
    # NIVEL 2: Procesamiento por bin — paralelo o secuencial
    # =========================================================================

    if n_workers > 1:
        # ── Modo paralelo ────────────────────────────────────────────────────
        log(f"\n⚙️  Modo paralelo: {n_workers} workers.")

        # Fase 1: pre-embeber todos los papers (GPU secuencial)
        if prefetch:
            from fronts.parallel import prefetch_embeddings
            prefetch_embeddings(subfield_name, df_windows,
                                batch_size=SPECTER_BATCH_SIZE, verbose=verbose)

        # Fase 2: bins en paralelo
        from fronts.parallel import run_bins_parallel
        df_all = run_bins_parallel(
            subfield_name, df_windows,
            force_from=force_from, n_workers=n_workers, verbose=verbose
        )

        # Fase 3: tracking longitudinal
        if not df_all.empty and 'cluster_leiden' in df_all.columns:
            tracking_path = sub_dir / "transitions.parquet"
            if _should_force(force_from, 'tracking') or not tracking_path.exists():
                log("\n📊 Jaccard tracking longitudinal...")
                from fronts.longitudinal.cluster_tracker import track_clusters
                df_tracking = track_clusters(df_all, 'cluster_leiden', 'bin_id')
                _save_parquet(df_tracking, tracking_path)

        final_path = sub_dir / "fronts_result.parquet"
        _save_parquet(df_all, final_path)
        log(f"\n✅ Pipeline paralelo completo: {len(df_all):,} papers. → {final_path}")
        return df_all

    # ── Modo secuencial (n_workers == 1) ─────────────────────────────────────
    log(f"\n⚙️  Modo secuencial (n_workers=1).")
    all_results = []

    for _, win in df_windows.iterrows():
        bin_id = int(win['bin_id'])
        y_start, y_end = int(win['y_start']), int(win['y_end'])
        win_dir = get_window_cache_dir(sub_clean, bin_id)
        log(f"\n── Bin {bin_id:03d} [{y_start}-{y_end}] ({win['mode']}) ──")

        # ── Nivel 1b: Pares de citas ─────────────────────────────────────────
        cit_path = win_dir / "citations.parquet"
        df_cit = None if _should_force(force_from, 'citations') else _load_parquet(cit_path)
        if df_cit is None:
            log("   Extrayendo pares de citas de ClickHouse...")
            df_cit = get_citation_pairs(subfield_name, y_start, y_end)
            _save_parquet(df_cit, cit_path)
            log(f"   ✅ {len(df_cit):,} pares de citas.")
        else:
            log(f"   📦 Citas desde cache: {len(df_cit):,} pares.")

        # ── Nivel 2a: Metadata del bin para embeddings/topológico ─────────────
        meta_path = win_dir / "metadata.parquet"
        df_meta = None if _should_force(force_from, 'citations') else _load_parquet(meta_path)
        if df_meta is None:
            from fronts.clickhouse_queries import get_bin_metadata
            df_meta = get_bin_metadata(subfield_name, y_start, y_end)
            _save_parquet(df_meta, meta_path)

        if df_meta is None or df_meta.empty:
            log(f"   ⚠️  Sin metadata para bin {bin_id}, saltando.")
            continue

        n_papers = len(df_meta)

        # ── Nivel 2b: Leiden (Estructural) ────────────────────────────────────
        leiden_path = win_dir / "structural" / "leiden.parquet"
        df_leiden = None if _should_force(force_from, 'structural') else _load_parquet(leiden_path)
        if df_leiden is None:
            log("   🔗 Construyendo matriz de acoplamiento...")
            # Añadir referenced_works a metadata para build_citation_matrix
            if 'referenced_works' in df_meta.columns:
                C_BC, work_to_idx = build_citation_matrix(df_meta, open_corpus=OPEN_CORPUS)
            else:
                C_BC, work_to_idx = None, {}

            if C_BC is not None:
                S = normalize_salton(C_BC)
                S = apply_salton_threshold(S, SALTON_THRESHOLD)
                clusters = run_leiden(S, resolution=LEIDEN_RESOLUTION)
                idx_to_id = {v: k for k, v in work_to_idx.items()}
                df_leiden = pd.DataFrame({
                    'id': [idx_to_id[i] for i in range(len(clusters))],
                    'cluster_leiden': clusters
                })
            else:
                df_leiden = pd.DataFrame({'id': df_meta['id'].tolist(),
                                          'cluster_leiden': [-1] * n_papers})
            _save_parquet(df_leiden, leiden_path)
            log(f"   ✅ Leiden: {df_leiden['cluster_leiden'].nunique()} clusters.")
        else:
            log(f"   📦 Leiden desde cache: {df_leiden['cluster_leiden'].nunique()} clusters.")

        # ── Nivel 2c: UMAP ────────────────────────────────────────────────────
        umap_path = win_dir / "semantic" / "umap30d.npy"
        ids_path = win_dir / "semantic" / "ids.npy"
        projections = None if _should_force(force_from, 'umap') else _load_npy(umap_path)

        if projections is None:
            log("   🔤 Generando/recuperando embeddings SPECTER2...")
            from fronts.embeddings.cache_manager import (
                get_missing_ids, get_embeddings_for_window, insert_embeddings
            )
            # Recuperar embeddings del cache
            df_emb = get_embeddings_for_window(subfield_name, y_start, y_end, 'embedding_specter2')
            missing_ids = get_missing_ids(subfield_name, y_start, y_end, 'embedding_specter2')

            if missing_ids:
                log(f"   ⚡ Generando {len(missing_ids):,} embeddings nuevos (SPECTER2)...")
                df_new = df_meta[df_meta['id'].isin(missing_ids)][['id', 'title', 'abstract']]
                texts = prepare_text_for_specter(df_new)
                new_embs = generate_specter_embeddings(texts, batch_size=SPECTER_BATCH_SIZE)
                insert_embeddings(
                    df_new['id'].tolist(), new_embs, subfield_name,
                    df_new.get('publication_year', [y_start] * len(df_new)).tolist()
                )
                df_emb = get_embeddings_for_window(subfield_name, y_start, y_end, 'embedding_specter2')

            if df_emb.empty:
                log("   ⚠️  Sin embeddings para este bin, saltando UMAP.")
                continue

            paper_ids_ordered = df_emb['id'].tolist()
            embeddings = np.vstack(df_emb['embedding'].tolist()).astype(np.float32)

            log("   📐 UMAP 768d → 30d...")
            projections = reduce_dimensions(
                embeddings, n_components=UMAP_N_COMPONENTS,
                n_neighbors=UMAP_N_NEIGHBORS, metric=UMAP_METRIC, min_dist=UMAP_MIN_DIST
            )
            _save_npy(projections, umap_path)
            _save_npy(np.array(paper_ids_ordered, dtype=object), ids_path)
        else:
            paper_ids_ordered = _load_npy(ids_path).tolist()
            log(f"   📦 UMAP desde cache: {projections.shape}.")

        # ── Nivel 2d: HDBSCAN Semántico ───────────────────────────────────────
        hdb_sem_path = win_dir / "semantic" / "hdbscan.parquet"
        df_hdb_sem = None if _should_force(force_from, 'semantic') else _load_parquet(hdb_sem_path)
        if df_hdb_sem is None:
            log("   🌐 HDBSCAN semántico...")
            sem_labels, _ = run_hdbscan(
                projections,
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric=HDBSCAN_METRIC
            )
            df_hdb_sem = pd.DataFrame({'id': paper_ids_ordered, 'cluster_semantic': sem_labels})
            _save_parquet(df_hdb_sem, hdb_sem_path)
        else:
            log(f"   📦 HDBSCAN semántico desde cache: {df_hdb_sem['cluster_semantic'].nunique()} clusters.")

        # ── Nivel 2e: FastRP Topológico ───────────────────────────────────────
        fastrp_path = win_dir / "topological" / "fastrp.npy"
        hdb_top_path = win_dir / "topological" / "hdbscan.parquet"
        df_hdb_top = None if _should_force(force_from, 'topological') else _load_parquet(hdb_top_path)

        if df_hdb_top is None:
            fastrp_vecs = _load_npy(fastrp_path) if not _should_force(force_from, 'topological') else None
            if fastrp_vecs is None:
                log("   🕸️  Construyendo grafo heterogéneo (igraph)...")
                g, work_ids = build_heterogeneous_graph(df_cit, df_meta)
                if g.vcount() > 0:
                    log(f"   ⚡ FastRP {FASTRP_DIMENSION}d ({FASTRP_ITERATIONS} iter)...")
                    fastrp_vecs = run_fastrp_igraph_sparse(
                        g, len(work_ids), FASTRP_DIMENSION, FASTRP_ITERATIONS
                    )
                    _save_npy(fastrp_vecs, fastrp_path)
                else:
                    fastrp_vecs = np.zeros((n_papers, FASTRP_DIMENSION), dtype=np.float32)

            log("   🌐 HDBSCAN topológico...")
            top_labels, _ = run_hdbscan(
                fastrp_vecs,
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric='cosine'
            )
            # Alinear IDs con los del grafo
            g_ids = df_meta['id'].tolist()[:fastrp_vecs.shape[0]]
            df_hdb_top = pd.DataFrame({'id': g_ids, 'cluster_topological': top_labels})
            _save_parquet(df_hdb_top, hdb_top_path)
        else:
            log(f"   📦 Topológico desde cache: {df_hdb_top['cluster_topological'].nunique()} clusters.")

        # ── Nivel 3a: AMI por bin ─────────────────────────────────────────────
        ami_path = win_dir / "ami.parquet"
        df_ami = None if _should_force(force_from, 'ami') else _load_parquet(ami_path)
        if df_ami is None:
            df_merged = df_meta[['id']].copy()
            for df_cl, col in [(df_leiden, 'cluster_leiden'),
                               (df_hdb_sem, 'cluster_semantic'),
                               (df_hdb_top, 'cluster_topological')]:
                if df_cl is not None and col in df_cl.columns:
                    df_merged = df_merged.merge(df_cl[['id', col]], on='id', how='left')

            ami_records = {}
            pairs = [('cluster_leiden', 'cluster_semantic'),
                     ('cluster_leiden', 'cluster_topological'),
                     ('cluster_semantic', 'cluster_topological')]
            for c1, c2 in pairs:
                if c1 in df_merged.columns and c2 in df_merged.columns:
                    valid = df_merged[[c1, c2]].dropna()
                    valid = valid[(valid[c1] != -1) & (valid[c2] != -1)]
                    if len(valid) > 10:
                        ami = adjusted_mutual_info_score(valid[c1], valid[c2])
                        ami_records[f'ami_{c1}_vs_{c2}'] = ami
                        log(f"   AMI {c1} vs {c2}: {ami:.4f}")

            ami_records['bin_id'] = bin_id
            ami_records['y_start'] = y_start
            ami_records['y_end'] = y_end
            df_ami = pd.DataFrame([ami_records])
            _save_parquet(df_ami, ami_path)

        # Consolidar resultado del bin
        df_bin_result = df_meta[['id', 'publication_year']].copy()
        df_bin_result['bin_id'] = bin_id
        df_bin_result['y_start'] = y_start
        df_bin_result['y_end'] = y_end
        df_bin_result['bin_mode'] = win['mode']

        for df_cl, col in [(df_leiden, 'cluster_leiden'),
                           (df_hdb_sem, 'cluster_semantic'),
                           (df_hdb_top, 'cluster_topological')]:
            if df_cl is not None and col in df_cl.columns:
                df_bin_result = df_bin_result.merge(df_cl[['id', col]], on='id', how='left')

        all_results.append(df_bin_result)

    if not all_results:
        return pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True)

    # =========================================================================
    # NIVEL 3B: Tracking longitudinal (Jaccard)
    # =========================================================================
    tracking_path = sub_dir / "transitions.parquet"
    df_tracking = None if _should_force(force_from, 'tracking') else _load_parquet(tracking_path)
    if df_tracking is None and 'cluster_leiden' in df_all.columns:
        log("\n📊 Nivel 3b: Jaccard tracking longitudinal...")
        df_tracking = track_clusters(df_all, 'cluster_leiden', 'bin_id')
        _save_parquet(df_tracking, tracking_path)
        log(f"   ✅ {len(df_tracking):,} transiciones entre bins.")

    # Guardar resultado final
    final_path = sub_dir / "fronts_result.parquet"
    _save_parquet(df_all, final_path)
    log(f"\n✅ Pipeline completo. {len(df_all):,} papers procesados. Guardado en {final_path}")
    return df_all


def get_consistency_metrics(df: pd.DataFrame) -> dict:
    """Retorna métricas AMI entre métodos para el DataFrame consolidado."""
    metrics = {}
    pairs = [('cluster_leiden', 'cluster_semantic'),
             ('cluster_leiden', 'cluster_topological'),
             ('cluster_semantic', 'cluster_topological')]
    for c1, c2 in pairs:
        if c1 in df.columns and c2 in df.columns:
            valid = df[[c1, c2]].dropna()
            valid = valid[(valid[c1] != -1) & (valid[c2] != -1)]
            if len(valid) > 10:
                metrics[f'AMI ({c1} vs {c2})'] = adjusted_mutual_info_score(valid[c1], valid[c2])
    return metrics
