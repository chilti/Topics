# Plan de Implementación Maestro: Detección Longitudinal de Frentes de Investigación (v4.0)

> **Directorio de Trabajo:** `revistaslatam/fronts/`  
> **Estrategia:** Triple Pipeline Multimodal (Estructural, Semántico, Topológico)  
> **Hardware:** Optimizado para GPU RTX 4090 (SPECTER2) y ClickHouse (Matrices Sparse).

---

## 1. Resumen del Pipeline Multimodal

El sistema detectará frentes de investigación mediante tres enfoques que permiten detectar tanto áreas consolidadas como "silos" de conocimiento emergente:

1.  **Estructural (Red de Citas):** Basado en acoplamiento bibliográfico ($C_{BC}$) y algoritmo de Leiden.
2.  **Semántico (Contenido):** Basado en embeddings SPECTER2 y clustering HDBSCAN.
3.  **Topológico (Ecosistema):** Basado en FastRP (Neo4j GDS) para capturar cercanía por autores/instituciones compartidas.

---

## 2. Arquitectura del Directorio `fronts/`

```
revistaslatam/
└── fronts/
    ├── __init__.py                    # Exports públicos
    ├── config.py                      # Parámetros globales (k, γ, etc.)
    ├── pipeline.py                    # Orquestador (triple detección)
    ├── clickhouse_queries.py          # SQL centralizado (vía ClickHouse client)
    ├── visualizations.py             # Alluvial + Triple Network View + UMAP
    │
    ├── segmentation/
    │   └── temporal_bins.py          # Vigintiles de volumen constante
    │
    ├── structural/
    │   ├── citation_network.py       # Matrices A, C_BC (scipy.sparse)
    │   └── leiden_detector.py        # Leiden + Salton normalization
    │
    ├── semantic/
    │   ├── embeddings.py             # SPECTER2 inference (Transformers)
    │   ├── dimensionality.py         # UMAP (768d -> 5d)
    │   └── hdbscan_detector.py       # HDBSCAN clustering
    │
    ├── topological/                   # NUEVO: FastRP (vía Neo4j/GDS)
    │   └── fastrp_detector.py        # FastRP + HDBSCAN
    │
    ├── longitudinal/
    │   ├── cluster_tracker.py        # Jaccard tracking
    │   ├── ami_consistency.py        # AMI entre particiones
    │   └── event_detector.py         # Fisión, Fusión, Emergencia, Extinción
    │
    └── labeling/
        ├── tfidf_extractor.py        # TF-IDF sobre abstracts por clúster
        └── llm_namer.py              # Naming con LLM Local (LM Studio)
```

---

## 3. Módulo por Módulo: Especificación Detallada

### Módulo 0: `config.py`
```python
K_BINS = 20                    # Vigintiles
MIN_COUPLING_WEIGHT = 3        # Mínimo de referencias compartidas
LEIDEN_RESOLUTION = 1.0        # γ: resolución de Leiden
FASTRP_DIM = 128               # Dimensiones de proyección topológica
UMAP_N_COMPONENTS = 5          # Reducción para HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 30  # Tamaño mínimo de frente
JACCARD_THRESHOLD = 0.3        # Umbral de persistencia temporal
```

### Módulo 1: `1_segmentation/temporal_bins.py`
Normaliza la Ley de Price ($N(t) = N_0 e^{rt}$) dividiendo el corpus en $k$ ventanas de volumen constante $V = N_{total}/k$.
- **Algoritmo:** `tj = np.quantile(years, [j/k for j in range(1, k+1)], method='lower')`.
- **SQL:** `SELECT publication_year FROM works WHERE arrayExists(x -> x = {subfield_id}, subfields_ids)`.

### Módulo 2: `2_structural/`
- **Matrices:** Construcción de matriz binaria **A** (Artículos x Referencias).
- **Acoplamiento ($C_{BC}$):** `C_BC = A * A.T`. Captura papers actuales que citan el mismo pasado.
- **Normalización (Salton):** $S_{ik} = C_{ik} / \sqrt{k_i \cdot k_k}$. Neutraliza sesgos por bibliografías extensas.
- **Pseudocódigo Leiden:**
  ```python
  graph = ig.Graph.Weighted_Adjacency(S_ik.tolist(), mode="undirected")
  partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0)
  ```

### Módulo 3: `3_semantic/`
- **SPECTER2:** `allenai/specter2_base`. Input: `Title: [t] [SEP] Abstract: [a]`.
- **Batching:** `batch_size=2000` (para RTX 4090).
- **UMAP:** `n_neighbors=15, metric='cosine'`.
- **HDBSCAN:** `cluster_selection_method='eom'` (maximiza estabilidad).

### Módulo 4: `4_topological/`
- **Neo4j GDS:** Proyectar grafo `Work -> Author`, `Work -> Institution`, `Work -> Source`.
- **FastRP:** Generar vectores de 128d. Captura cercanía por colaboración e infraestructura científica.
- **HDBSCAN:** Aplicado sobre los vectores de FastRP.

### Módulo 5: `5_longitudinal/`
- **Tracking:** $J(C_i, C_j) = |C_i \cap C_j| / |C_i \cup C_j|$.
- **AMI (Consistency):** Medir coincidencia entre los 3 métodos.
  - **Silo:** $AMI(Sem, Struc) \to 0$ (Hablan de lo mismo pero no se citan).
  - **Consolidado:** $AMI \to 1$.

### Módulo 6: `6_labeling/`
- **TF-IDF:** Extraer términos clave de los abstracts de cada cluster.
- **LLM Local (LM Studio):** Patrón `lib/llm_utils.py`.
- **Prompt:** "Genera un nombre técnico descriptivo de máximo 10 palabras en español para este frente basado en términos: {terms} y títulos: {titles}."

---

## 4. `clickhouse_queries.py` — Queries de Referencia

```sql
-- 1. Pares de citas para matriz A
SELECT w.work_id AS source, ref_id AS target
FROM works w ARRAY JOIN w.referenced_works AS ref_id
WHERE arrayExists(x -> x = {subfield_id}, w.subfields_ids)
  AND w.publication_year BETWEEN {y_start} AND {y_end}

-- 2. Metadata para caracterización
SELECT work_id, title, abstract, citation_count, fwci
FROM works WHERE work_id IN {ids_cluster}
```

---

## 5. `visualizations.py` — Visualización Premium

1. **Diagrama de Aluvión (go.Sankey):** Evolución temporal. Color de cintas basado en AMI (Verde=Consolidado, Rojo=Silo).
2. **Vista Triple de Redes:**
   - **Estructural:** Citación (Pyvis/NetworkX).
   - **Semántica:** UMAP (Plotly Scatter 2D).
   - **Topológica:** Ecosistema (Pyvis).
3. **Métricas de Impacto:** Bubble charts de frentes (Eje X: Edad, Eje Y: FWCI, Tamaño: Docs).

---

## 6. Integración en `dashboard_tema.py`

- **Botón:** En la sidebar, sustituyendo el checkbox.
- **Persistencia:** Almacenamiento en `data/cache_temas/fronts_{id}.parquet`.
- **Carga:** `if st.session_state.get('run_fronts'): ...`.

---

## 7. Plan de Fases y Prioridades

| Fase | Contenido | Prioridad |
|------|-----------|-----------|
| **1** | Bins Temporales y Queries ClickHouse | 🔴 ALTA |
| **2** | Leiden (Estructural) | 🔴 ALTA |
| **3** | FastRP (Topológico) y SPECTER2 (Semántico) | 🔴 ALTA |
| **4** | Tracking Jaccard y Consistencia AMI | 🟡 MEDIA |
| **5** | Etiquetado LLM Local y Visualizaciones Triples | 🟡 MEDIA |
| **6** | Integración Final Dashboard (Botón + Caché) | 🔴 ALTA |

---

## 8. Hardware y Dependencias
- **Inferencia:** `transformers`, `torch`, `sentence-transformers`.
- **Gráficos:** `leidenalg`, `igraph`, `networkx`.
- **Matemáticas:** `scipy`, `numpy`, `umap-learn`, `hdbscan`.
- **DB:** `clickhouse-connect`, `neo4j`.
