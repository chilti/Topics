import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def build_citation_matrix(df):
    """
    Construye la matriz de adyacencia A (Artículos x Referencias compartidas).
    Retorna la matriz de acoplamiento bibliográfico C_BC = A * A.T
    """
    # 1. Mapeo de IDs a índices
    unique_works = df['id'].unique()
    work_to_idx = {wid: i for i, wid in enumerate(unique_works)}
    num_works = len(unique_works)
    
    # 2. Recopilar todas las referencias únicas que están dentro de nuestro corpus
    all_refs = set()
    for refs in df['referenced_works']:
        for r in refs:
            # Limpiar comillas si vienen de JSONExtractArrayRaw
            r_clean = r.strip('"')
            if r_clean in work_to_idx: # Solo citas internas para este sandbox
                all_refs.add(r_clean)
    
    ref_to_idx = {rid: i for i, rid in enumerate(all_refs)}
    num_refs = len(all_refs)
    
    if num_refs == 0:
        return None, work_to_idx
    
    # 3. Construir matriz dispersa A
    rows = []
    cols = []
    data = []
    
    for i, row in df.iterrows():
        w_idx = work_to_idx[row['id']]
        for ref in row['referenced_works']:
            r_clean = ref.strip('"')
            if r_clean in ref_to_idx:
                rows.append(w_idx)
                cols.append(ref_to_idx[r_clean])
                data.append(1)
                
    A = csr_matrix((data, (rows, cols)), shape=(num_works, num_refs))
    
    # 4. Calcular Acoplamiento Bibliográfico (C_BC)
    C_BC = A @ A.T
    
    return C_BC, work_to_idx

def normalize_salton(C):
    """Aplica la normalización de Coseno de Salton a una matriz de adyacencia."""
    # Convertir a densa para el sandbox (pequeño), pero idealmente seguir en sparse
    C_dense = C.toarray()
    k = np.sqrt(np.diag(C_dense))
    
    # S_ik = C_ik / sqrt(k_i * k_k)
    # Evitar división por cero
    k_inv = np.where(k > 0, 1.0 / k, 0)
    S = C_dense * k_inv[:, np.newaxis] * k_inv[np.newaxis, :]
    
    return S
