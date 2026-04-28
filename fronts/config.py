# Parámetros globales para la detección de frentes de investigación (v5.0)
# Sincronizado con fronts/PLAN.md y docs/critica_academica_fronts.md

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Segmentación Temporal
# ---------------------------------------------------------------------------
K_BINS = 20           # Vigintiles: análisis histórico completo
WINDOW_YEARS = 3      # Ventana deslizante para análisis reciente (2010+)
WINDOW_STEP = 1       # Paso en años para la ventana deslizante
RECENT_FROM = 2010    # Año de inicio para análisis con ventana deslizante

# ---------------------------------------------------------------------------
# 2. Detección Estructural (Leiden + Acoplamiento Bibliográfico)
# ---------------------------------------------------------------------------
SALTON_THRESHOLD = 0.1    # Umbral coseno de Salton (Waltman 2016) — scale-free
                           # Reemplaza MIN_COUPLING_WEIGHT absoluto (era 3)
LEIDEN_RESOLUTION = 1.0   # γ: resolución de Leiden. Evaluar sensibilidad 0.5–2.0
OPEN_CORPUS = True         # True: incluye referencias externas al subcampo
                           # Captura frentes interdisciplinares (Glänzel & Thijs 2012)

# ---------------------------------------------------------------------------
# 3. Detección Semántica (SPECTER2 → UMAP → HDBSCAN)
# ---------------------------------------------------------------------------
SPECTER_MODEL = "allenai/specter2_base"
SPECTER_BATCH_SIZE = 256      # Óptimo para RTX 4090 (24GB VRAM)

UMAP_N_COMPONENTS = 30        # 768d → 30d (no 5d como antes)
UMAP_N_NEIGHBORS = 30         # Más vecinos para subcampos grandes
UMAP_METRIC = 'cosine'        # Preserva geometría no lineal de transformers
UMAP_MIN_DIST = 0.0           # 0.0 maximiza densidad local para HDBSCAN

HDBSCAN_MIN_CLUSTER_SIZE = 50  # Tamaño mínimo de frente (ajustar por bin)
HDBSCAN_MIN_SAMPLES = 10       # Muestras mínimas para núcleo denso
HDBSCAN_METRIC = 'cosine'      # Coseno: discriminativo hasta ~100d
HDBSCAN_METHOD = 'eom'         # Excess of Mass: más estable que 'leaf'

# ---------------------------------------------------------------------------
# 4. Detección Topológica (FastRP via igraph/cuGraph)
# ---------------------------------------------------------------------------
FASTRP_DIMENSION = 128    # Dimensiones de la proyección FastRP
FASTRP_ITERATIONS = 3     # Iteraciones de propagación (L=3)
TOPOLOGICAL_BACKEND = 'igraph'  # 'igraph' (CPU) o 'cugraph' (RTX 4090)

# ---------------------------------------------------------------------------
# 5. Tracking Longitudinal y Consistencia
# ---------------------------------------------------------------------------
JACCARD_THRESHOLD = 0.3   # Umbral para considerar "mismo frente" entre bins
JACCARD_MIN = 0.05        # Umbral mínimo para registrar transición

# ---------------------------------------------------------------------------
# 6. Etiquetado (TF-IDF + LLM Local)
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 1000
TOP_TERMS_PER_CLUSTER = 10
TOP_TITLES_PER_CLUSTER = 5

# ---------------------------------------------------------------------------
# 7. Paralelismo
# ---------------------------------------------------------------------------
import os

# Cores físicos disponibles (puede sobreescribirse en runtime)
N_PHYSICAL_CORES = os.cpu_count() or 4

# Workers para procesar bins en paralelo (ProcessPoolExecutor).
# Regla: N_BIN_WORKERS = min(n_bins, n_physical_cores // 2)
# Dejar la mitad de cores libres para BLAS/numpy dentro de cada worker.
N_BIN_WORKERS = max(1, N_PHYSICAL_CORES // 2)

# Workers para paralelizar structural + topológico dentro de un bin.
# False = secuencial (menos memoria); True = paralelo (más velocidad).
WITHIN_BIN_PARALLEL = True

# Threads que UMAP usa internamente.
# None = usa random_state=42 (reproducible, single-thread).
# -1   = usa todos los cores (no reproducible, más rápido).
UMAP_N_JOBS: int | None = None  # Cambiar a -1 para máxima velocidad

# Threads que scipy/numpy/BLAS usan para álgebra lineal.
# Se setea como variable de entorno antes de importar numpy.
# None = BLAS elige automáticamente (generalmente todos los cores).
BLAS_NUM_THREADS: int | None = None  # None = auto

# ---------------------------------------------------------------------------
# 7. Checkpoints y Rutas de Cache
# ---------------------------------------------------------------------------
CACHE_ROOT = Path("data/cache_fronts")

def get_window_cache_dir(subfield_clean: str, bin_id: int) -> Path:
    d = CACHE_ROOT / subfield_clean / f"window_{bin_id:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_subfield_cache_dir(subfield_clean: str) -> Path:
    d = CACHE_ROOT / subfield_clean
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------------------------------------------------------
# Niveles de pipeline (para force_from)
# ---------------------------------------------------------------------------
PIPELINE_LEVELS = [
    'windows',      # Nivel 1: particionamiento en bins
    'citations',    # Nivel 1: extracción de pares de citas de ClickHouse
    'structural',   # Nivel 2: Leiden por bin
    'umap',         # Nivel 2: UMAP 768→30d
    'semantic',     # Nivel 2: HDBSCAN semántico
    'topological',  # Nivel 2: FastRP + HDBSCAN topológico
    'ami',          # Nivel 3: AMI + matriz de contingencia
    'tracking',     # Nivel 3: Jaccard tracking entre bins
    'labeling',     # Nivel 4: etiquetado LLM
]
