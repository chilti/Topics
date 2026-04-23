# Parámetros globales para la detección de frentes de investigación

# 1. Segmentación Temporal
K_BINS = 20                    # Número de ventanas de volumen constante (vigintiles)

# 2. Detección Estructural (Leiden)
MIN_COUPLING_WEIGHT = 3        # Mínimo de referencias compartidas para considerar un enlace
LEIDEN_RESOLUTION = 1.0        # Parámetro gamma (gamma=1.0 es modularidad estándar)

# 3. Detección Semántica (SPECTER2 + HDBSCAN)
SPECTER_MODEL = "allenai/specter2_base"
UMAP_N_COMPONENTS = 5          # Dimensiones reducidas para clustering
UMAP_N_NEIGHBORS = 15          # Vecinos para construcción de grafo UMAP
HDBSCAN_MIN_CLUSTER_SIZE = 30  # Tamaño mínimo de un frente detectado
HDBSCAN_MIN_SAMPLES = 5        # Muestras mínimas para núcleo denso

# 4. Detección Topológica (FastRP)
NEO4J_URI = "bolt://localhost:7687"
FASTRP_DIMENSION = 128         # Dimensiones de la proyección FastRP
FASTRP_ITERATIONS = 3          # Iteraciones del algoritmo FastRP

# 5. Tracking y Consistencia
JACCARD_THRESHOLD = 0.3        # Umbral para considerar "mismo frente" en el tiempo

# 6. Etiquetado (LLM)
TFIDF_MAX_FEATURES = 1000
TOP_TERMS_PER_CLUSTER = 10
TOP_TITLES_PER_CLUSTER = 5
