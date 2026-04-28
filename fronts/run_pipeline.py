"""
fronts/run_pipeline.py — Script de entrada principal para el pipeline de frentes.

ORDEN CORRECTO DE EJECUCIÓN:
  Paso 1: Crear tabla embeddings_cache en ClickHouse (una sola vez)
  Paso 2: Generar embeddings SPECTER2 (GPU, puede tomar horas)
  Paso 3: Correr el pipeline completo (bins paralelos)

Uso:
    # Paso 1: crear tabla (no hace nada si ya existe)
    python -m fronts.run_pipeline --step setup --subfield "Pulmonary and Respiratory Medicine"

    # Paso 2: generar embeddings (requiere GPU para velocidad)
    python -m fronts.run_pipeline --step embed --subfield "Pulmonary and Respiratory Medicine"

    # Paso 3: pipeline completo
    python -m fronts.run_pipeline --step run --subfield "Pulmonary and Respiratory Medicine"

    # Todo en secuencia (1→2→3)
    python -m fronts.run_pipeline --step all --subfield "Pulmonary and Respiratory Medicine"

    # Re-correr solo desde HDBSCAN (conserva UMAP y embeddings cacheados)
    python -m fronts.run_pipeline --step run --subfield "..." --force-from semantic

    # Ver cobertura actual de embeddings
    python -m fronts.run_pipeline --step status --subfield "Pulmonary and Respiratory Medicine"
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Paso 1: Setup — crear tabla embeddings_cache
# ---------------------------------------------------------------------------

def step_setup(args):
    print("\n" + "─" * 55)
    print("  PASO 1: Crear tabla embeddings_cache en ClickHouse")
    print("─" * 55)

    from fronts.embeddings.cache_manager import ensure_table_exists
    ensure_table_exists()
    print("   Tabla lista. Puedes continuar con --step embed.")


# ---------------------------------------------------------------------------
# Paso 2: Embed — generar embeddings SPECTER2 para un subcampo
# ---------------------------------------------------------------------------

def step_embed(args):
    print("\n" + "─" * 55)
    print(f"  PASO 2: Generar embeddings SPECTER2")
    print(f"  Subcampo : {args.subfield}")
    print(f"  Modo     : {args.mode}")
    print("─" * 55)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n   Dispositivo: {device}")
    if device == "cpu":
        print("   Sin GPU. La generacion sera LENTA (~20ms/paper en CPU vs ~2ms en GPU).")
        # Solo pedir confirmacion si la salida no está redirigida a un archivo (como hace el dashboard)
        if sys.stdout.isatty():
            try:
                resp = input("   Continuar de todas formas? [s/N]: ").strip().lower()
                if resp != 's':
                    print("   Cancelado.")
                    return
            except EOFError:
                print("   Continuando (EOF detectado).")
        else:
            print("   Continuando (subproceso con stdout redirigido).")

    import numpy as np
    import pandas as pd
    from fronts.config import (
        K_BINS, WINDOW_YEARS, WINDOW_STEP, RECENT_FROM, SPECTER_BATCH_SIZE
    )
    from fronts.segmentation.temporal_bins import (
        compute_temporal_bins, compute_sliding_windows
    )
    from fronts.clickhouse_queries import get_years_for_subfield, get_bin_metadata
    from fronts.embeddings.cache_manager import (
        ensure_table_exists, get_missing_ids, insert_embeddings, get_coverage_report
    )
    from fronts.semantic.embeddings import generate_specter_embeddings, prepare_text_for_specter

    # Asegurar que la tabla existe
    ensure_table_exists()

    # Calcular ventanas según el modo
    print("\n   Consultando distribución de años...")
    years = get_years_for_subfield(args.subfield)
    if not years:
        print(f"   ❌ No hay datos para '{args.subfield}'")
        return

    year_min, year_max = min(years), max(years)
    print(f"   Años: {year_min}–{year_max}  |  Total papers: {len(years):,}")

    if args.mode == 'vigintiles':
        windows = compute_temporal_bins(years, k=K_BINS)
    elif args.mode == 'sliding':
        windows = compute_sliding_windows(
            max(RECENT_FROM, year_min), year_max, WINDOW_YEARS, WINDOW_STEP
        )
    else:  # both
        windows  = compute_temporal_bins(years, k=K_BINS)
        windows += compute_sliding_windows(
            max(RECENT_FROM, year_min), year_max, WINDOW_YEARS, WINDOW_STEP
        )
    windows = list(set(windows))  # Deduplicar

    print(f"   Ventanas a cubrir: {len(windows)}")

    # Identificar papers que faltan embeber en TODO el rango
    global_y_start = min(w[0] for w in windows)
    global_y_end   = max(w[1] for w in windows)
    print(f"\n   Buscando papers sin embedding ({global_y_start}–{global_y_end})...")
    missing = get_missing_ids(args.subfield, global_y_start, global_y_end, 'embedding_specter2')

    if not missing:
        print("   ✅ Todos los papers ya tienen embeddings en cache. Nada que hacer.")
        step_status(args)
        return

    print(f"   Papers a embeber: {len(missing):,}")
    print(f"   Batch size: {SPECTER_BATCH_SIZE}  |  Dispositivo: {device}")
    tiempo_est = len(missing) * (2 if device == 'cuda' else 20) / 1000 / 60
    print(f"   Tiempo estimado: ~{tiempo_est:.0f} min")

    # Obtener metadata de los papers que faltan
    print("\n   Descargando metadata de ClickHouse...")
    df_meta = get_bin_metadata(args.subfield, global_y_start, global_y_end)
    df_new  = df_meta[df_meta['id'].isin(set(missing))][['id', 'title', 'abstract', 'publication_year']]
    print(f"   Papers descargados: {len(df_new):,}")

    if df_new.empty:
        print("   ❌ Sin metadata para los papers. Revisar ClickHouse.")
        return

    # Generar embeddings
    print("\n   Preparando textos Title+Abstract...")
    texts = prepare_text_for_specter(df_new)
    print(f"   Generando {len(texts):,} embeddings SPECTER2...")
    embeddings = generate_specter_embeddings(
        texts,
        model_name='allenai/specter2_base',
        batch_size=SPECTER_BATCH_SIZE
    )
    print(f"   Shape generado: {embeddings.shape}")

    # Insertar en ClickHouse
    print("\n   Insertando en embeddings_cache...")
    insert_embeddings(
        df_new['id'].tolist(),
        embeddings,
        args.subfield,
        df_new['publication_year'].tolist(),
        model_col='embedding_specter2'
    )

    step_status(args)


# ---------------------------------------------------------------------------
# Paso 3: Run — ejecutar el pipeline de clustering
# ---------------------------------------------------------------------------

def step_run(args):
    print("\n" + "─" * 55)
    print(f"  PASO 3: Pipeline de clustering")
    print(f"  Subcampo  : {args.subfield}")
    print(f"  Modo      : {args.mode}")
    print(f"  Workers   : {args.workers}")
    print(f"  Force-from: {args.force_from or 'ninguno (usar cache)'}")
    print("─" * 55)

    from fronts.pipeline import run_fronts_analysis

    df = run_fronts_analysis(
        subfield_name=args.subfield,
        mode=args.mode,
        force_from=args.force_from,
        n_workers=args.workers,
        prefetch=False,   # Embeddings ya generados en paso 2
        verbose=True
    )

    if df.empty:
        print("\nEl pipeline no produjo resultados.")
    else:
        import datetime
        print(f"\nPipeline completado: {len(df):,} papers procesados.")
        print(f"   Columnas: {list(df.columns)}")
        sub_clean = args.subfield.strip().lower().replace(' ', '_')
        out_path = f"data/cache_fronts/{sub_clean}/fronts_result.parquet"
        print(f"   Resultados en: {out_path}")
        # Escribir sentinel .done para que el dashboard sepa que terminó
        done_path = Path(f"data/cache_fronts/{sub_clean}/.done")
        done_path.write_text(datetime.datetime.now().isoformat())
        print(f"   Sentinel .done escrito en: {done_path}")


# ---------------------------------------------------------------------------
# Status: reporte de cobertura de embeddings
# ---------------------------------------------------------------------------

def step_status(args):
    print("\n" + "─" * 55)
    print(f"  STATUS: Cobertura de embeddings")
    print(f"  Subcampo: {args.subfield}")
    print("─" * 55)

    from fronts.embeddings.cache_manager import get_coverage_report
    df = get_coverage_report(args.subfield)

    if df.empty:
        print("   Sin datos en embeddings_cache para este subcampo.")
        return

    row = df.iloc[0]
    total = int(row.get('n_total', 0))
    print(f"\n   Total papers en cache : {total:,}")
    print(f"   SPECTER2              : {int(row.get('n_specter2', 0)):,}  ({_pct(row, 'n_specter2', total)})")
    print(f"   SciBERT               : {int(row.get('n_scilbert', 0)):,}  ({_pct(row, 'n_scilbert', total)})")
    print(f"   FastRP-citas          : {int(row.get('n_fastrp_cit', 0)):,}  ({_pct(row, 'n_fastrp_cit', total)})")
    print(f"   FastRP-heterogéneo    : {int(row.get('n_fastrp_het', 0)):,}  ({_pct(row, 'n_fastrp_het', total)})")
    print(f"   UMAP-30d              : {int(row.get('n_umap_30d', 0)):,}  ({_pct(row, 'n_umap_30d', total)})")
    last = row.get('last_update', 'N/A')
    print(f"   Última actualización  : {last}")


def _pct(row, col, total):
    if total == 0:
        return "—"
    return f"{int(row.get(col, 0)) / total * 100:.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de frentes de investigación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m fronts.run_pipeline --step setup --subfield "Pulmonary and Respiratory Medicine"
  python -m fronts.run_pipeline --step embed --subfield "Pulmonary and Respiratory Medicine"
  python -m fronts.run_pipeline --step run   --subfield "Pulmonary and Respiratory Medicine"
  python -m fronts.run_pipeline --step all   --subfield "Pulmonary and Respiratory Medicine" --workers 4
  python -m fronts.run_pipeline --step status --subfield "Pulmonary and Respiratory Medicine"

  # Re-correr solo el clustering desde HDBSCAN (conserva UMAP):
  python -m fronts.run_pipeline --step run --subfield "..." --force-from semantic

  # Ventanas deslizantes de 3 años (análisis reciente):
  python -m fronts.run_pipeline --step all --subfield "..." --mode sliding
        """
    )
    parser.add_argument('--step', required=True,
                        choices=['setup', 'embed', 'run', 'all', 'status'],
                        help='Paso a ejecutar')
    parser.add_argument('--subfield', required=True,
                        help='Nombre del subcampo OpenAlex')
    parser.add_argument('--mode', default='sliding',
                        choices=['vigintiles', 'sliding', 'both'],
                        help='Estrategia de segmentación temporal (default: sliding)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Procesos paralelos (default: N_PHYSICAL_CORES // 2; 1=secuencial)')
    parser.add_argument('--force-from', dest='force_from', default=None,
                        choices=['windows', 'citations', 'structural', 'umap',
                                 'semantic', 'topological', 'ami', 'tracking', 'labeling'],
                        help='Re-ejecutar desde este nivel (ignora cache desde aquí)')

    args = parser.parse_args()

    try:
        if args.step == 'setup':
            step_setup(args)
        elif args.step == 'embed':
            step_embed(args)
        elif args.step == 'run':
            step_run(args)
        elif args.step == 'status':
            step_status(args)
        elif args.step == 'all':
            step_setup(args)
            step_embed(args)
            step_run(args)
    except Exception as e:
        import traceback
        err_str = str(e)
        if "HTTPConnectionPool" in err_str or "ConnectTimeoutError" in err_str or "OperationalError" in err_str:
            print("\n❌ ERROR CRÍTICO DE CONEXIÓN A CLICKHOUSE")
            print("No se pudo establecer conexión con el servidor de base de datos.")
            print("Causas comunes:")
            print("  1. El servidor ClickHouse está apagado o reiniciándose.")
            print("  2. La conexión VPN se desconectó.")
            print("  3. La IP o puerto en el archivo .env son incorrectos o inaccesibles.")
            print(f"\nDetalle técnico:\n{err_str}")
        else:
            print("\n❌ ERROR INESPERADO EN EL PIPELINE:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
