"""
segmentation/temporal_bins.py
Dos estrategias de segmentación temporal complementarias:
  1. Vigintiles de volumen constante (análisis histórico).
  2. Ventanas deslizantes de anchura fija (análisis reciente, alta resolución).
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Vigintiles
# ---------------------------------------------------------------------------

def compute_temporal_bins(years, k: int = 20) -> List[Tuple[int, int]]:
    """
    Calcula k ventanas de volumen constante (cuantiles de la distribución de años).
    Normaliza la Ley de Price: N(t) = N0 * e^(rt).

    Args:
        years: Lista o array de años de publicación del corpus completo.
        k: Número de bins (20 = vigintiles).

    Returns:
        Lista de tuplas (año_inicio, año_fin) sin bins vacíos.
    """
    if not len(years):
        return []

    years_sorted = np.sort(years)
    quantile_probs = [j / k for j in range(1, k + 1)]
    cutpoints = np.quantile(years_sorted, quantile_probs, method='lower')

    bins = []
    prev = int(years_sorted[0])
    unique_cuts = sorted(set(int(c) for c in cutpoints))

    for cut in unique_cuts:
        bins.append((prev, cut))
        prev = cut + 1

    # Garantizar que el último bin cubre el año máximo real
    if bins:
        bins[-1] = (bins[-1][0], int(years_sorted[-1]))

    return bins


# ---------------------------------------------------------------------------
# Ventanas deslizantes
# ---------------------------------------------------------------------------

def compute_sliding_windows(
    year_min: int,
    year_max: int,
    window_size: int = 3,
    step: int = 1
) -> List[Tuple[int, int]]:
    """
    Genera ventanas deslizantes de anchura fija para análisis reciente.
    Ejemplo: window_size=3, step=1 → (2010,2012),(2011,2013),...,(2023,2025).

    Args:
        year_min: Primer año de inicio (ej. 2010).
        year_max: Último año del corpus.
        window_size: Anchura de la ventana en años.
        step: Paso en años entre ventanas consecutivas.

    Returns:
        Lista de tuplas (año_inicio, año_fin).
    """
    windows = []
    start = year_min
    while start + window_size - 1 <= year_max:
        windows.append((start, start + window_size - 1))
        start += step
    return windows


# ---------------------------------------------------------------------------
# Asignación de papers a bins
# ---------------------------------------------------------------------------

def assign_bin(year: int, bins: List[Tuple[int, int]]) -> int:
    """Asigna un año a su bin correspondiente. Retorna -1 si no encaja."""
    for i, (start, end) in enumerate(bins):
        if start <= year <= end:
            return i
    return -1


def assign_bins_vectorized(
    years: np.ndarray,
    bins: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Asignación vectorizada de años a bins (mucho más rápida que assign_bin en loop).

    Returns:
        Array de enteros con el índice de bin para cada año (-1 = fuera de rango).
    """
    result = np.full(len(years), -1, dtype=np.int32)
    for i, (start, end) in enumerate(bins):
        mask = (years >= start) & (years <= end)
        result[mask] = i
    return result
