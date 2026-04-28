# Plan de Implementación Maestro: Detección Longitudinal de Frentes de Investigación (v5.0)

> **Directorio de Trabajo:** `Topics/fronts/`  
> **Estrategia:** Triple Pipeline Multimodal (Estructural, Semántico, Topológico)  
> **Hardware:** RTX 4090 (SPECTER2/cuGraph), ClickHouse 1.5TB, servidor Neo4j 128GB RAM.  
> **Última revisión:** 2026-04-27 — ver `docs/critica_academica_fronts.md`

---

## 1. Resumen del Pipeline Multimodal

El sistema detectará frentes de investigación mediante tres enfoques que permiten detectar tanto áreas consolidadas como "silos" de conocimiento emergente:

1.  **Estructural (Red de Citas):** Acoplamiento bibliográfico ($C_{BC}$, corpus abierto) + Leiden por bin temporal, umbral de Salton ≥ 0.1.
2.  **Semántico (Contenido):** SPECTER2 → UMAP(768→30d, coseno) → HDBSCAN(coseno). Embeddings cacheados en ClickHouse.
3.  **Topológico (Ecosistema):** FastRP sobre grafo heterogéneo (citas + autores + instituciones + revista) vía **igraph** (primario) o **cuGraph** (GPU). Una instancia por ventana temporal, sin bloqueo de Neo4j GDS.

---

## 2. Arquitectura del Directorio `fronts/`

```
Topics/
└── fronts/
    ├── __init__.py                    # Exports públicos
    ├── config.py                      # Parámetros globales (k, γ, etc.)
    ├── pipeline.py                    # Orquestador (triple detección)
    ├── clickhouse_queries.py          # SQL centralizado
    ├── visualizations.py             # Alluvial + Triple Network View + UMAP
    │
    ├── docs/                          # Documentación académica y diseño
    │   └── critica_academica_fronts.md  # Crítica y justificaciones metodológicas
    │
    ├── segmentation/
    │   └── temporal_bins.py          # Vigintiles + ventanas deslizantes
    │
    ├── structural/
    │   ├── citation_network.py       # Matrices A, C_BC (scipy.sparse, corpus abierto)
    │   └── leiden_detector.py        # Leiden + Salton threshold ≥ 0.1
    │
    ├── semantic/
    │   ├── embeddings.py             # SPECTER2 inference, batch_size=256
    │   ├── dimensionality.py         # UMAP 768d→30d, metric='cosine'
    │   └── hdbscan_detector.py       # HDBSCAN metric='cosine'
    │
    ├── embeddings/                    # NUEVO: gestión del cache en ClickHouse
    │   └── cache_manager.py          # DDL, insert, get_missing_ids, get_for_window
    │
    ├── topological/                   # FastRP vía igraph/cuGraph (sin Neo4j GDS)
    │   └── fastrp_detector.py        # FastRP sobre grafo heterogéneo por ventana
    │
    ├── longitudinal/
    │   ├── cluster_tracker.py        # Jaccard tracking (sets pre-calculados)
    │   ├── ami_consistency.py        # AMI + matriz de contingencia cluster-a-cluster
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
K_BINS = 20                    # Vigintiles (análisis histórico)
WINDOW_YEARS = 3               # Ventana deslizante para análisis reciente (2010+)
SALTON_THRESHOLD = 0.1        # Umbral coseno de Salton (Waltman 2016), reemplaza MIN_COUPLING_WEIGHT absoluto
LEIDEN_RESOLUTION = 1.0        # γ: resolución de Leiden (evaluar sensibilidad 0.5–2.0)
FASTRP_DIM = 128               # Dimensiones de proyección topológica
UMAP_N_COMPONENTS = 30         # Reducción para HDBSCAN (768d → 30d, métrica coseno)
UMAP_N_NEIGHBORS = 30          # Más vecinos que el default para subcampos grandes
HDBSCAN_MIN_CLUSTER_SIZE = 50  # Tamaño mínimo de frente (ajustar por bin)
HDBSCAN_METRIC = 'cosine'      # Coseno: funciona bien en alta dimensionalidad
JACCARD_THRESHOLD = 0.3        # Umbral de persistencia temporal
EMBEDDING_CACHE_PATH = 'data/cache_fronts/embeddings_cache.parquet'  # Cache por paper ID
```

### Módulo 1: `segmentation/temporal_bins.py`
Normaliza la Ley de Price ($N(t) = N_0 e^{rt}$) mediante dos estrategias complementarias:

**1a. Vigintiles de volumen constante** (análisis histórico completo):
- **Algoritmo:** `tj = np.quantile(years, [j/k for j in range(1, k+1)], method='lower')`.
- Garantiza mismo número de papers por ventana, independientemente del año calendario.

**1b. Ventanas deslizantes de 3 años** (análisis reciente de alta resolución, 2010-presente):
- Para subcampos con crecimiento explosivo reciente (ej. Pulmonary: ~451K papers en 2025 solo).
- Ventana de 3 años ≈ 1.2M papers, viable en hardware actual (ClickHouse 1.5TB, Neo4j con 100GB libres).
- Paso de 1 año produce 13 ventanas superpuestas (2010-2012, 2011-2013, ..., 2023-2025).
- Uso: detectar emergencia y extinción de frentes con mayor resolubilidad temporal.

### Módulo 2: `structural/`
- **Matrices:** Construcción de matriz binaria **A** (Artículos × Referencias). Se construye **por separado para cada bin temporal** (no sobre el corpus completo) para evitar mezclar comunidades de décadas distintas.
- **Referencias:** Se incluyen **todas** las obras referenciadas (internas y externas al subcampo) para capturar frentes interdisciplinares. El espacio de columnas de A es el universo de referencias citadas, no solo los papers del subcampo.
- **Acoplamiento ($C_{BC}$):** `C_BC = A @ A.T`. Captura papers actuales que citan el mismo pasado (incluyendo papers de otros campos).
- **Normalización (Salton):** $S_{ik} = C_{ik} / \sqrt{k_i \cdot k_k}$. Neutraliza sesgos por bibliografías extensas.
- **Umbral de aristas:** Se aplica umbral $S_{ik} \geq 0.1$ (Waltman, 2016) sobre la matriz de cosenos normalizada, reemplazando el umbral de peso absoluto. Es scale-free: funciona igual en biomedicina y humanidades.
- **Pseudocódigo Leiden (por bin):**
  ```python
  for bin_start, bin_end in temporal_bins:
      df_bin = df[(df['year'] >= bin_start) & (df['year'] <= bin_end)]
      C_BC, _ = build_citation_matrix(df_bin)    # Matriz del bin
      S = normalize_salton(C_BC)
      S = np.where(S >= SALTON_THRESHOLD, S, 0)  # Filtrar aristas débiles
      clusters = run_leiden(S, resolution=LEIDEN_RESOLUTION)
  ```

### Módulo 3: `semantic/`
- **SPECTER2:** `allenai/specter2_base`. Input: `Title: [t] [SEP] Abstract: [a]`.
- **Batching:** `batch_size=256` (para RTX 4090, maximiza VRAM de 24GB).
- **Cache de embeddings:** Guardar `(id, embedding[768])` en Parquet por paper ID. Solo se embeben papers nuevos por ventana (~150K/año), evitando re-embeber el corpus completo en cada ventana. 6M embeddings float32 ≈ 18GB en disco.
- **Reducción dimensional:** UMAP directamente de 768d → 30d con `metric='cosine'` y `n_neighbors=30`. La proyección no lineal de UMAP es superior a PCA para embeddings de transformers.
- **HDBSCAN:** `cluster_selection_method='eom'`, `metric='cosine'`. La métrica coseno mantiene discriminabilidad en 30 dimensiones (McInnes & Healy, 2018).
- **Pipeline:**
  ```python
  # 1. Recuperar embeddings del cache o generar nuevos
  new_ids = [id for id in df['id'] if id not in embedding_cache]
  new_embeddings = generate_specter_embeddings(df[df['id'].isin(new_ids)])
  # 2. UMAP: 768d -> 30d (no lineal, coseno)
  projections = umap.UMAP(n_components=30, metric='cosine', n_neighbors=30).fit_transform(all_embeddings)
  # 3. HDBSCAN con coseno
  labels = hdbscan.HDBSCAN(min_cluster_size=50, metric='cosine').fit_predict(projections)
  ```

### Módulo 4: `topological/`
- **Backend:** `igraph` (primario, CPU) o `cuGraph` (secundario, RTX 4090). Reemplaza Neo4j GDS que solo permite una proyección activa.
- **Unidad de análisis:** ventana temporal individual, no el subcampo completo (6M papers). Cada ventana se carga, procesa y descarta.
- **Red heterogénea** (diferenciador clave frente al estructural):
  - `Work → Work` (citas directas)
  - `Work → Author` (co-autoría)
  - `Work → Institution` (afiliación)
  - `Work → Source` (revista)
- **FastRP manual (igraph):**
  ```python
  g = ig.Graph(n=num_nodes, edges=all_edges, directed=False)
  A = g.get_adjacency_sparse()
  R = np.random.randn(A.shape[1], FASTRP_DIM) / np.sqrt(FASTRP_DIM)
  # Propagación L=3 iteraciones
  H = A @ R
  for _ in range(2): H = A @ H + R
  embeddings = H  # (n_nodes, 128)
  ```
- **Viabilidad memoria (ventana 3 años, 1.2M papers):** igraph ≈ 5-8 GB RAM, dentro del margen con 100 GB libres.
- **HDBSCAN:** `metric='cosine'` sobre los vectores de FastRP.

### Módulo 5: `longitudinal/`
- **Tracking:** $J(C_i, C_j) = |C_i \cap C_j| / |C_i \cup C_j|$. Sets pre-calculados antes del bucle doble.
- **AMI (Consistency):** Reportar AMI junto con el número de clusters de cada partición (el techo teórico depende de la cardinalidad).
  - **Evaluación independiente primero:** se evalúa cada método por separado. La fusión se decidirá en base a los resultados.
  - **Silo:** $AMI(Sem, Struc) \to 0$ — mismos papers, marcos referenciales distintos.
  - **Consolidado:** $AMI \to 1$.
- **Matriz de contingencia:** heat map normalizado de solapamiento cluster-a-cluster (complemento al AMI global).

### Módulo 6: `6_labeling/`
- **TF-IDF:** Extraer términos clave de los abstracts de cada cluster.
- **LLM Local (LM Studio):** Patrón `lib/llm_utils.py`.
- **Prompt:** "Genera un nombre técnico descriptivo de máximo 10 palabras en español para este frente basado en términos: {terms} y títulos: {titles}."

---

## 4. `embeddings_cache` — Tabla ClickHouse

Arquitectura de datos para embeddings: **wide-format** (una columna por modelo), separada de `works_flat`.

```sql
CREATE TABLE IF NOT EXISTS embeddings_cache (
    id                   String,
    subfield_name        LowCardinality(String),
    publication_year     UInt16,
    -- Semánticos
    embedding_specter2   Array(Float32) DEFAULT [],   -- 768d
    embedding_scilbert   Array(Float32) DEFAULT [],   -- 768d
    -- Topológicos
    embedding_fastrp_cit Array(Float32) DEFAULT [],   -- 128d, solo citas
    embedding_fastrp_het Array(Float32) DEFAULT [],   -- 128d, red heterogénea
    -- Proyecciones
    embedding_umap_30d   Array(Float32) DEFAULT [],   -- 30d
    -- Auditoría
    specter2_at  Nullable(DateTime),
    fastrp_het_at Nullable(DateTime),
    updated_at   DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (subfield_name, publication_year, id);

-- Agregar nuevo modelo (instantáneo, sin reescritura):
ALTER TABLE embeddings_cache ADD COLUMN embedding_bge_m3 Array(Float32) DEFAULT [];
```

**Principio:** `works_flat` nunca se modifica para agregar embeddings. Solo INSERTs en `embeddings_cache`.

## 5. `clickhouse_queries.py` — Queries de Referencia

```sql
-- 1. Pares de citas (corpus abierto: todas las referencias, no solo internas)
SELECT W.id AS source_id, ref AS target_id
FROM works_flat AS W
ARRAY JOIN W.referenced_works AS ref
INNER JOIN (SELECT id FROM works_flat WHERE subfield_name = '{subfield}') AS T ON ref = T.id
WHERE W.subfield_name = '{subfield}'
  AND W.publication_year BETWEEN {y_start} AND {y_end}

-- 2. Papers sin embedding (para saber qué hay que generar)
SELECT w.id
FROM works_flat AS w
LEFT JOIN (SELECT id FROM embeddings_cache WHERE length(embedding_specter2) > 0) AS e ON w.id = e.id
WHERE w.subfield_name = '{subfield}'
  AND w.publication_year BETWEEN {y_start} AND {y_end}
  AND e.id IS NULL

-- 3. Metadata para caracterización de clusters
SELECT id, title, abstract, cited_by_count, fwci
FROM works_flat WHERE id IN ({ids_cluster})
```

---

## 6. `visualizations.py` — Visualización Premium

1. **Diagrama de Aluón (go.Sankey):** Evolución temporal. Color de cintas basado en AMI (Verde=Consolidado, Rojo=Silo).
2. **Vista Triple de Redes:** Estructural (Pyvis), Semántica (UMAP Plotly 2D), Topológica (Pyvis).
3. **Matriz de contingencia AMI:** Heatmap de solapamiento cluster-a-cluster por par de métodos.
4. **Bubble charts de frentes:** Eje X: Edad, Eje Y: FWCI, Tamaño: Docs.

---

## 7. Integración en Dashboard

- **Botón:** En la sidebar de `dashboard_topics.py`.
- **Persistencia:** `data/cache_temas/fronts_{subfield}.parquet`.
- **Carga:** `if st.session_state.get('run_fronts'): ...`.

---

## 8. Plan de Fases y Prioridades

| Fase | Contenido | Prioridad |
|------|-----------|----------|
| **1** | `embeddings_cache` DDL + `cache_manager.py` | 🔴 ALTA |
| **2** | Bins temporales (vigintiles + ventanas 3 años) | 🔴 ALTA |
| **3** | Leiden por bin (corpus abierto, Salton ≥ 0.1) | 🔴 ALTA |
| **4** | SPECTER2 → UMAP(30d) → HDBSCAN(coseno) | 🔴 ALTA |
| **5** | FastRP heterogéneo vía igraph/cuGraph | 🔴 ALTA |
| **6** | Tracking Jaccard + AMI + matriz contingencia | 🟡 MEDIA |
| **7** | Etiquetado LLM Local y Visualizaciones Triples | 🟡 MEDIA |
| **8** | Integración Final Dashboard | 🔴 ALTA |

---

## 9. Hardware y Dependencias

- **Inferencia:** `transformers`, `torch`, `sentence-transformers`.
- **Grafos:** `leidenalg`, `igraph` (primario), `cugraph` (GPU secundario).
- **Matemáticas:** `scipy`, `numpy`, `umap-learn`, `hdbscan`.
- **DB:** `clickhouse-connect`.
- **Neo4j:** solo para almacenamiento permanente, no para compute de frentes.

---

## 10. Documentación

Ver `docs/critica_academica_fronts.md` para justificaciones metodológicas detalladas,
referencias bibliográficas y decisiones de diseño acordadas.
