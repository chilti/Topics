"""
fronts/tests/smoke_test.py
Test de humo para verificar que todos los módulos del pipeline funcionen.

Modos:
  - SINTÉTICO (default): genera datos fake, no requiere ClickHouse ni GPU.
    Verifica que todas las funciones corren sin errores.
  - REAL (--real): usa un query pequeño de ClickHouse (200 papers COVID).
    Verifica integración completa incluyendo cache y pipeline.py.

Uso:
    cd c:\\Users\\jlja\\Documents\\Proyectos\\Topics
    python -m fronts.tests.smoke_test             # modo sintético
    python -m fronts.tests.smoke_test --real      # modo real (requiere ClickHouse)
    python -m fronts.tests.smoke_test --module citation_network  # un módulo
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

# Asegurar que el root está en el path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Parámetros reducidos para el test (sobreescriben config.py)
# ---------------------------------------------------------------------------
TEST_N_PAPERS = 300          # Papers sintéticos
TEST_N_REFS = 500            # Pool total de referencias (internas + externas)
TEST_N_EXTERNAL = 200        # Referencias externas al subcampo
TEST_REFS_PER_PAPER = 8      # Referencias por paper
TEST_K_BINS = 3              # Bins temporales (vs 20 en producción)
TEST_MIN_CLUSTER = 5         # min_cluster_size HDBSCAN (vs 50 en producción)
TEST_UMAP_NEIGHBORS = 15     # n_neighbors UMAP (debe ser < n_papers)
TEST_EMBED_DIM = 768         # Dimensión de embeddings simulados
TEST_YEARS = list(range(2015, 2024))  # 2015-2023
TEST_SUBFIELD = "Smoke Test Subfield"

# Colores para la consola
OK   = "✅"
FAIL = "❌"
SKIP = "⏭️ "
INFO = "ℹ️ "


# ---------------------------------------------------------------------------
# Generador de datos sintéticos
# ---------------------------------------------------------------------------

def make_synthetic_papers(n=TEST_N_PAPERS):
    """Genera un DataFrame de papers con referencias aleatorias."""
    np.random.seed(42)
    internal_ids = [f"W{i:06d}" for i in range(n)]
    external_ids = [f"EXT{i:06d}" for i in range(TEST_N_EXTERNAL)]
    all_ref_pool = internal_ids + external_ids

    author_pool = [f"A{i:04d}" for i in range(50)]
    inst_pool   = [f"I{i:03d}" for i in range(20)]
    src_pool    = [f"S{i:02d}" for i in range(10)]

    rows = []
    for wid in internal_ids:
        n_refs = np.random.randint(3, TEST_REFS_PER_PAPER + 1)
        refs = list(np.random.choice(all_ref_pool, n_refs, replace=False))
        rows.append({
            'id': wid,
            'title': f"Paper {wid}: a study of X",
            'abstract': f"We studied X in the context of Y. Results show Z. {wid}",
            'publication_year': int(np.random.choice(TEST_YEARS)),
            'cited_by_count': int(np.random.randint(0, 200)),
            'fwci': float(np.random.exponential(1.2)),
            'referenced_works': refs,
            'institution_ids': list(np.random.choice(inst_pool, 2, replace=False)),
            'source_id': str(np.random.choice(src_pool)),
            'author_ids': list(np.random.choice(author_pool, np.random.randint(1, 5), replace=False)),
        })
    return pd.DataFrame(rows)


def make_synthetic_citation_pairs(df_papers):
    """Genera pares de citas a partir del DataFrame de papers."""
    rows = []
    for _, row in df_papers.iterrows():
        for ref in row['referenced_works']:
            if ref in set(df_papers['id']):  # Solo citas internas para este helper
                rows.append({'source_id': row['id'], 'target_id': ref})
    return pd.DataFrame(rows)


def make_synthetic_embeddings(n=TEST_N_PAPERS, dim=TEST_EMBED_DIM):
    """Genera embeddings aleatorios normalizados (simulan SPECTER2)."""
    np.random.seed(42)
    emb = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms > 0, norms, 1.0)


# ---------------------------------------------------------------------------
# Tests por módulo
# ---------------------------------------------------------------------------

def test_temporal_bins(df_papers):
    from fronts.segmentation.temporal_bins import (
        compute_temporal_bins, compute_sliding_windows, assign_bins_vectorized
    )
    years = df_papers['publication_year'].values

    bins = compute_temporal_bins(years, k=TEST_K_BINS)
    assert len(bins) > 0, "No se generaron bins"
    assert bins[-1][1] == int(years.max()), "El último bin no cubre el año máximo"

    sliding = compute_sliding_windows(2015, 2023, window_size=3, step=1)
    assert len(sliding) > 0

    assignments = assign_bins_vectorized(years, bins)
    assert len(assignments) == len(years)
    assert (assignments >= -1).all()

    print(f"  vigintiles={len(bins)}, sliding={len(sliding)}, assigned={np.sum(assignments >= 0)}/{len(years)}")
    return bins


def test_citation_network(df_papers):
    from fronts.structural.citation_network import (
        build_citation_matrix, normalize_salton, apply_salton_threshold
    )
    # Corpus abierto
    C_BC, work_to_idx = build_citation_matrix(df_papers, open_corpus=True)
    assert C_BC is not None, "C_BC es None con corpus abierto"
    assert C_BC.shape[0] == len(df_papers)

    S = normalize_salton(C_BC)
    assert S.shape == (len(df_papers), len(df_papers))
    assert S.max() <= 1.0 + 1e-6, f"Salton > 1.0: {S.max()}"

    S_thresh = apply_salton_threshold(S, threshold=0.1)
    n_edges = np.count_nonzero(S_thresh) // 2
    print(f"  C_BC shape={C_BC.shape}, aristas tras umbral={n_edges}")
    return S_thresh, work_to_idx


def test_leiden(S_matrix):
    from fronts.structural.leiden_detector import run_leiden
    clusters = run_leiden(S_matrix, resolution=1.0)
    assert len(clusters) == S_matrix.shape[0]
    n_clusters = len(set(clusters))
    assert n_clusters >= 1
    print(f"  Leiden: {n_clusters} clusters en {len(clusters)} nodos")
    return clusters


def test_umap(embeddings):
    from fronts.semantic.dimensionality import reduce_dimensions
    projections = reduce_dimensions(
        embeddings,
        n_components=5,            # 5 para el test (velocidad); producción usa 30
        n_neighbors=TEST_UMAP_NEIGHBORS,
        metric='cosine',
        min_dist=0.0
    )
    assert projections.shape == (len(embeddings), 5)
    assert not np.any(np.isnan(projections)), "UMAP produjo NaN"
    print(f"  UMAP: {embeddings.shape} → {projections.shape}")
    return projections


def test_hdbscan(projections):
    from fronts.semantic.hdbscan_detector import run_hdbscan
    labels, clusterer = run_hdbscan(
        projections,
        min_cluster_size=TEST_MIN_CLUSTER,
        min_samples=3,
        metric='cosine'
    )
    assert len(labels) == len(projections)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 0
    print(f"  HDBSCAN: {n_clusters} clusters, ruido={int(np.sum(labels == -1))}")
    return labels


def test_fastrp(df_papers, df_cit):
    from fronts.topological.fastrp_detector import (
        build_heterogeneous_graph, run_fastrp_igraph_sparse
    )
    g, work_ids = build_heterogeneous_graph(
        df_cit, df_papers,
        include_authors=True, include_institutions=True, include_sources=True
    )
    assert g.vcount() > 0
    assert len(work_ids) == len(df_papers)

    embs = run_fastrp_igraph_sparse(g, len(work_ids), embedding_dim=32, n_iterations=2)
    assert embs.shape == (len(work_ids), 32)
    assert not np.any(np.isnan(embs)), "FastRP produjo NaN"
    print(f"  igraph: {g.vcount()} nodos, {g.ecount()} aristas. FastRP: {embs.shape}")
    return embs


def test_ami(leiden_labels, hdb_labels):
    from sklearn.metrics import adjusted_mutual_info_score
    valid = [(l, h) for l, h in zip(leiden_labels, hdb_labels[:len(leiden_labels)])
             if l != -1 and h != -1]
    if len(valid) < 5:
        print("  ⚠️  Muy pocos puntos válidos para AMI")
        return 0.0
    l_arr = [v[0] for v in valid]
    h_arr = [v[1] for v in valid]
    ami = adjusted_mutual_info_score(l_arr, h_arr)
    print(f"  AMI (Leiden vs HDBSCAN): {ami:.4f}  [n={len(valid)}]")
    return ami


def test_cache_manager():
    """Prueba DDL y operaciones básicas de embeddings_cache."""
    from fronts.embeddings.cache_manager import ensure_table_exists, get_coverage_report
    ensure_table_exists()
    df_cov = get_coverage_report(TEST_SUBFIELD)
    print(f"  embeddings_cache: tabla lista, cobertura={df_cov.to_dict(orient='records')}")


def test_parallel_bins(df_papers, df_cit):
    """Verifica que _bin_worker se puede ejecutar en ProcessPoolExecutor."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fronts.parallel import _bin_worker
    from fronts.segmentation.temporal_bins import compute_temporal_bins

    years = df_papers['publication_year'].values
    bins = compute_temporal_bins(years, k=2)

    jobs = []
    for i, (y_start, y_end) in enumerate(bins[:2]):  # Solo 2 bins para el test
        jobs.append({
            'subfield_name': TEST_SUBFIELD,
            'sub_clean': 'smoke_test_subfield',
            'bin_id': i,
            'y_start': y_start,
            'y_end': y_end,
            'force_levels': {'windows', 'citations', 'structural', 'umap',
                             'semantic', 'topological', 'ami'},
            'root_path': str(ROOT),
        })

    # Verificar que los jobs son picklables
    import pickle
    for job in jobs:
        pickle.dumps(job)  # Si falla aquí, el worker no puede serializarse

    print(f"  Jobs picklables: {len(jobs)}")
    print(f"  CPUs disponibles: {os.cpu_count()}")
    # No lanzamos ProcessPoolExecutor real en el test sintético
    # (requeriría ClickHouse). Solo verificamos la serialización.
    return True


def test_pipeline_real(subfield_name: str, n_workers: int = 1):
    """
    Test de integración real: usa run_fronts_analysis con parámetros reducidos.
    Requiere ClickHouse activo.
    """
    from fronts import config as cfg

    # Sobreescribir parámetros de config para el test
    cfg.K_BINS = 2
    cfg.WINDOW_YEARS = 2
    cfg.HDBSCAN_MIN_CLUSTER_SIZE = 5
    cfg.HDBSCAN_MIN_SAMPLES = 2
    cfg.UMAP_N_COMPONENTS = 5
    cfg.UMAP_N_NEIGHBORS = 10
    cfg.FASTRP_DIMENSION = 32
    cfg.FASTRP_ITERATIONS = 2
    cfg.SPECTER_BATCH_SIZE = 16

    from fronts.pipeline import run_fronts_analysis
    df = run_fronts_analysis(
        subfield_name,
        mode='sliding',
        force_from='windows',
        n_workers=n_workers,
        prefetch=False,   # En el test no hay GPU
        verbose=True
    )
    assert isinstance(df, pd.DataFrame), "Pipeline no retornó DataFrame"
    print(f"  Pipeline ({n_workers} worker(s)): {len(df):,} papers, columnas={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_module_test(name, fn, *args):
    try:
        result = fn(*args)
        print(f"{OK} {name}")
        return result, True
    except Exception as e:
        print(f"{FAIL} {name}")
        traceback.print_exc()
        return None, False


def main():
    parser = argparse.ArgumentParser(description="Smoke test del pipeline de frentes.")
    parser.add_argument('--real', action='store_true',
                        help='Ejecutar test de integración real con ClickHouse')
    parser.add_argument('--subfield', type=str,
                        default='Pulmonary and Respiratory Medicine',
                        help='Subcampo para el test real')
    parser.add_argument('--module', type=str, default=None,
                        help='Correr solo un módulo específico')
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  🧪 SMOKE TEST — Pipeline de Frentes de Investigación")
    print("═" * 60)

    results = {}

    # ── Datos sintéticos ──────────────────────────────────────────────────────
    print("\n📦 Generando datos sintéticos...")
    df_papers = make_synthetic_papers(TEST_N_PAPERS)
    df_cit    = make_synthetic_citation_pairs(df_papers)
    embeddings = make_synthetic_embeddings(TEST_N_PAPERS)
    print(f"   {len(df_papers)} papers, {len(df_cit)} pares de citas, "
          f"embeddings shape={embeddings.shape}")

    # ── Tests de módulos ──────────────────────────────────────────────────────
    if args.module in (None, 'temporal_bins'):
        print("\n🗓️  temporal_bins")
        bins, ok = run_module_test("temporal_bins", test_temporal_bins, df_papers)
        results['temporal_bins'] = ok

    if args.module in (None, 'citation_network'):
        print("\n🔗 citation_network")
        res, ok = run_module_test("citation_network", test_citation_network, df_papers)
        results['citation_network'] = ok
        S_matrix = res[0] if ok else None
        work_to_idx = res[1] if ok else {}

    if args.module in (None, 'leiden') and results.get('citation_network'):
        print("\n🌐 leiden_detector")
        leiden_labels, ok = run_module_test("leiden_detector", test_leiden, S_matrix)
        results['leiden'] = ok
    else:
        leiden_labels = None

    if args.module in (None, 'umap'):
        print("\n📐 dimensionality (UMAP)")
        projections, ok = run_module_test("umap", test_umap, embeddings)
        results['umap'] = ok

    if args.module in (None, 'hdbscan') and results.get('umap', True):
        if projections is None:
            projections, _ = run_module_test("umap_fallback", test_umap, embeddings)
        print("\n🌀 hdbscan_detector")
        hdb_labels, ok = run_module_test("hdbscan", test_hdbscan, projections)
        results['hdbscan'] = ok
    else:
        hdb_labels = None

    if args.module in (None, 'fastrp'):
        print("\n🕸️  fastrp_detector (igraph)")
        fastrp_embs, ok = run_module_test("fastrp", test_fastrp, df_papers, df_cit)
        results['fastrp'] = ok

    if args.module in (None, 'ami') and leiden_labels is not None and hdb_labels is not None:
        print("\n📊 AMI consistency")
        _, ok = run_module_test("ami", test_ami, leiden_labels, hdb_labels)
        results['ami'] = ok

    if args.module in (None, 'parallel'):
        print("\n⚙️  Paralelismo (pickling de workers)")
        _, ok = run_module_test("parallel_bins", test_parallel_bins, df_papers, df_cit)
        results['parallel_bins'] = ok

    # ── Test de integración real (opcional) ───────────────────────────────────
    if args.real:
        print(f"\n🔌 Test real con ClickHouse — '{args.subfield}'")
        try:
            from fronts.clickhouse_queries import get_ch_client
            client = get_ch_client()
            client.command("SELECT 1")
            print(f"   {OK} Conexión ClickHouse OK")

            print("\n   embeddings_cache DDL")
            _, ok = run_module_test("cache_manager", test_cache_manager)
            results['cache_manager'] = ok

            print("\n   Pipeline secuencial (n_workers=1)")
            _, ok = run_module_test(
                "pipeline_seq", test_pipeline_real, args.subfield, 1
            )
            results['pipeline_seq'] = ok

            print(f"\n   Pipeline paralelo (n_workers=2)")
            _, ok = run_module_test(
                "pipeline_parallel", test_pipeline_real, args.subfield, 2
            )
            results['pipeline_parallel'] = ok

        except Exception as e:
            print(f"   {FAIL} ClickHouse no disponible: {e}")
            results['pipeline_real'] = False
    else:
        print(f"\n{SKIP} Test real omitido (usar --real para activar)")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESUMEN")
    print("═" * 60)
    passed = sum(1 for v in results.values() if v is True)
    total  = len(results)
    for name, ok in results.items():
        icon = OK if ok else FAIL
        print(f"  {icon} {name}")
    print(f"\n  Total: {passed}/{total} módulos OK")
    print("═" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
