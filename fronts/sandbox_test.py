import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from fronts.clickhouse_queries import get_sandbox_data
from fronts.config import K_BINS

def main():
    print("--- INICIANDO SANDBOX TEST: Frentes de Investigacion ---")
    print("-" * 50)
    
    # 1. Extracción de Datos
    try:
        print("Paso 1: Extrayendo datos de ClickHouse (Subfield: 2737, Term: covid)...")
        df = get_sandbox_data(limit=1000)
        
        # Eliminar duplicados de ID si los hay
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        print(f"OK: Se obtuvieron {len(df)} papers unicos.")
        
        # Guardar localmente para no repetir la query
        os.makedirs("data/cache_fronts", exist_ok=True)
        df.to_parquet("data/cache_fronts/sandbox_raw.parquet", index=False)
        
    except Exception as e:
        print(f"ERROR en la extraccion: {e}")
        return

    # 2. Resumen de los datos
    print("\nResumen del corpus de prueba:")
    print(f"   - Anios: {df['publication_year'].min()} a {df['publication_year'].max()}")
    print(f"   - Papers con abstract: {df['abstract'].notna().sum()}")
    print(f"   - Papers con citas (internas): {df['referenced_works'].apply(len).sum()}")

    print("\nOK: Sandbox Step 1 Finalizado. Pasando al Step 2 (Estructural)...")
    print("-" * 50)

    # 2. Detección Estructural
    from fronts.structural.citation_network import build_citation_matrix, normalize_salton
    from fronts.structural.leiden_detector import run_leiden

    try:
        print("Paso 2: Construyendo matriz de citacion...")
        C_BC, work_to_idx = build_citation_matrix(df)
        
        if C_BC is None:
            print("AVISO: No hay citas internas suficientes para el analisis estructural.")
        else:
            print(f"OK: Matriz C_BC generada ({C_BC.shape[0]}x{C_BC.shape[1]})")
            
            print("Paso 2.1: Normalizando con Coseno de Salton...")
            S = normalize_salton(C_BC)
            
            print("Paso 2.2: Ejecutando algoritmo de Leiden...")
            clusters = run_leiden(S, resolution=1.0)
            df['cluster_leiden'] = clusters
            
            num_clusters = len(set(clusters))
            print(f"OK: Se detectaron {num_clusters} frentes estructurales.")

    except Exception as e:
        print(f"ERROR en el analisis estructural: {e}")

    # 3. Detección Semántica
    from fronts.semantic.embeddings import prepare_text_for_specter, generate_specter_embeddings
    from fronts.semantic.dimensionality import reduce_dimensions
    from fronts.semantic.hdbscan_detector import run_hdbscan

    try:
        cache_step3 = "data/cache_fronts/sandbox_results_step3.parquet"
        if os.path.exists(cache_step3):
            print("Paso 3: Cargando resultados semanticos desde cache...")
            df_cache = pd.read_parquet(cache_step3)
            if set(df_cache['id']) == set(df['id']):
                df['cluster_semantic'] = df_cache['cluster_semantic']
                print("OK: Cache semantica cargada.")
        
        if 'cluster_semantic' not in df.columns:
            print("Paso 3: Preparando textos para SPECTER2...")
            texts = prepare_text_for_specter(df)
            print("Paso 3.1: Generando embeddings (GPU)...")
            embeddings = generate_specter_embeddings(texts, batch_size=16)
            print("Paso 3.2: Reduciendo dimensionalidad con UMAP...")
            projections = reduce_dimensions(embeddings, n_components=5)
            print("Paso 3.3: Ejecutando HDBSCAN...")
            sem_clusters, _ = run_hdbscan(projections, min_cluster_size=10, min_samples=3)
            df['cluster_semantic'] = sem_clusters
            df.to_parquet(cache_step3, index=False)
            print("OK: Resultados semanticos guardados en cache.")
            
        num_sem_clusters = len(set(df['cluster_semantic'])) - (1 if -1 in df['cluster_semantic'].values else 0)
        print(f"OK: Se detectaron {num_sem_clusters} frentes semanticos.")

    except Exception as e:
        print(f"ERROR en el analisis semantico: {e}")

    # 4. Detección Topológica (FastRP)
    from fronts.topological.fastrp_detector import ingest_sandbox_to_neo4j, TopologicalFrontsManager

    try:
        print("\nOK: Sandbox Step 3 Finalizado. Pasando al Step 4 (Topologico)...")
        print("-" * 50)
        
        print("Paso 4: Ingestando datos en Neo4j Local (Docker)...")
        ingest_sandbox_to_neo4j(df)
        
        print("Paso 4.1: Ejecutando FastRP...")
        manager = TopologicalFrontsManager()
        manager.project_sandbox_graph("sandbox_graph")
        fastrp_embeddings_dict = manager.run_fastrp("sandbox_graph")
        
        fastrp_vecs = np.array([fastrp_embeddings_dict.get(wid, np.zeros(128)) for wid in df['id']])
        
        print("Paso 4.2: Clustering HDBSCAN sobre FastRP...")
        top_clusters, _ = run_hdbscan(fastrp_vecs, min_cluster_size=10, min_samples=3)
        df['cluster_topological'] = top_clusters
        
        num_top_clusters = len(set(top_clusters)) - (1 if -1 in top_clusters else 0)
        print(f"OK: Se detectaron {num_top_clusters} frentes topologicos.")
        manager.close()

    except Exception as e:
        print(f"ERROR en el analisis topologico: {e}")

    # 5. Análisis de Consistencia (AMI)
    from sklearn.metrics import adjusted_mutual_info_score
    print("\n" + "="*50)
    print("RESUMEN DE CONSISTENCIA (AMI)")
    print("="*50)
    
    if 'cluster_leiden' in df.columns and 'cluster_semantic' in df.columns:
        ami_str_sem = adjusted_mutual_info_score(df['cluster_leiden'], df['cluster_semantic'])
        print(f"AMI (Estructural vs Semantico): {ami_str_sem:.4f}")
        
    if 'cluster_leiden' in df.columns and 'cluster_topological' in df.columns:
        ami_str_top = adjusted_mutual_info_score(df['cluster_leiden'], df['cluster_topological'])
        print(f"AMI (Estructural vs Topologico): {ami_str_top:.4f}")

    df.to_parquet("data/cache_fronts/sandbox_results.parquet", index=False)
    print("\n" + "="*50)
    print("SANDBOX COMPLETO FINALIZADO EXITOSAMENTE")
    print("="*50)

if __name__ == "__main__":
    main()
