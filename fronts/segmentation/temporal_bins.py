import numpy as np

def compute_temporal_bins(years, k=20):
    """
    Calcula ventanas temporales de volumen constante (Vigintiles).
    years: lista o array de años de publicación.
    k: número de bins.
    Retorna: lista de tuplas (anio_inicio, anio_fin).
    """
    if not years:
        return []
    
    years_sorted = np.sort(years)
    # Usamos cuantiles para asegurar volumen constante
    quantile_probs = [j / k for j in range(1, k + 1)]
    cutpoints = np.quantile(years_sorted, quantile_probs, method='lower')
    
    bins = []
    prev = int(years_sorted[0])
    
    # Aseguramos que los puntos de corte sean únicos para evitar bins vacíos
    unique_cuts = sorted(list(set(cutpoints)))
    
    for cut in unique_cuts:
        bins.append((prev, int(cut)))
        prev = int(cut) + 1
        
    return bins

def assign_bin(year, bins):
    """Asigna un año a su bin correspondiente."""
    for i, (start, end) in enumerate(bins):
        if start <= year <= end:
            return i
    return -1
