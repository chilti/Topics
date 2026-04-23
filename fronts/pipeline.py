import pandas as pd
import os
from pathlib import Path
from fronts.clickhouse_queries import get_years_for_subfield, get_citation_pairs, get_sandbox_data
from fronts.segmentation.temporal_bins import compute_temporal_bins

def run_fronts_analysis(subfield_name, force_recalc=False):
    """
    Orquestador principal para la detección de frentes de investigación.
    """
    # 1. Verificar Cache
    cache_path = Path(f"data/cache_temas/fronts_{subfield_name.lower().replace(' ', '_')}.parquet")
    if cache_path.exists() and not force_recalc:
        print(f"📦 Cargando frentes desde cache: {cache_path}")
        return pd.read_parquet(cache_path)
    
    # 2. Ejecutar Pipeline (Simulado para el sandbox por ahora)
    # Aquí iría la lógica real que llama a structural, semantic y topological
    print(f"🚀 Ejecutando pipeline completo para: {subfield_name}")
    
    # Por ahora devolvemos un DataFrame vacío o el del sandbox si coincide
    if "Pulmonary" in subfield_name:
        sandbox_path = Path("data/cache_fronts/sandbox_results.parquet")
        if sandbox_path.exists():
            return pd.read_parquet(sandbox_path)
            
    return pd.DataFrame()

def get_consistency_metrics(df):
    """Retorna métricas AMI entre métodos."""
    from sklearn.metrics import adjusted_mutual_info_score
    metrics = {}
    if 'cluster_leiden' in df.columns and 'cluster_semantic' in df.columns:
        metrics['AMI (Estruct-Semant)'] = adjusted_mutual_info_score(df['cluster_leiden'], df['cluster_semantic'])
    return metrics
