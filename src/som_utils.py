"""
Utility functions for Hexagonal SOM visualization using Plotly.
Adapted from user provided script.
"""
import numpy as np

def hex_center(r, c, s=1.0, orientation="pointy"):
    """
    Calcula el centro (x, y) de un hexágono en la grilla (r, c).
    
    Args:
        r: índice de fila
        c: índice de columna
        s: radio del hexágono (centro a vértice)
        orientation: "pointy" o "flat"
    
    Returns:
        (x, y): coordenadas del centro
    """
    if orientation == "pointy":
        # pointy-top (odd-r offset): separación horizontal = sqrt(3)*s; vertical = 1.5*s
        x = np.sqrt(3)*s * (c + 0.5*(r % 2))
        y = 1.5*s * r
    else:
        # flat-top (odd-c offset): separación horizontal = 1.5*s; vertical = sqrt(3)*s
        x = 1.5*s * c
        y = np.sqrt(3)*s * (r + 0.5*(c % 2))
    return x, y

def hex_polygon(xc, yc, s=1.0, orientation="pointy"):
    """
    Genera los vértices (x, y) del hexágono dado su centro.
    Retorna 7 puntos para cerrar el polígono.
    """
    if orientation == "pointy":
        # vértice arriba
        angles = np.deg2rad([90, 150, 210, 270, 330, 30, 90])
    else:
        # lado arriba (top plano)
        angles = np.deg2rad([0, 60, 120, 180, 240, 300, 0])
    return xc + s*np.cos(angles), yc + s*np.sin(angles)
