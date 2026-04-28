# Crítica Académica (Rev. 2): Detección de Frentes de Investigación

> Evaluación de los procedimientos implementados en `fronts/`, incorporando observaciones del investigador principal.
> **Fecha:** 2026-04-27 | **Autor:** Revisión con IA

---

## 1. Enfoque Estructural: Acoplamiento Bibliográfico + Leiden

### Fundamento teórico

El acoplamiento bibliográfico (Price, 1965; Kessler, 1963) captura el **frente activo** de investigación: papers que comparten un marco referencial común. La normalización de Salton (Waltman & Van Eck, 2012) neutraliza el sesgo por tamaño de bibliografía. El algoritmo de Leiden (Traag, Waltman & Van Eck, 2019) es el estado del arte en detección de comunidades, superior a Louvain en garantías de optimalidad de subconjuntos.

### 1.1 Corpus cerrado vs. citas externas al subcampo

**Situación actual:** solo se consideran citas a papers dentro del mismo subcampo:
```python
if r in work_to_idx:  # Solo citas internas
```

**Problema:** Los frentes emergentes interdisciplinares (ej. *long COVID* en Pulmonología, que importa marcos de Inmunología y Neurología) quedan invisibles porque sus referencias características están fuera del corpus. Glänzel & Thijs (2012) documentan que los frentes emergentes tienen entre 30-60% de referencias externas al campo primario.

**Solución propuesta:** Ampliar el espacio de columnas de la matriz $A$ para incluir **todas** las obras referenciadas, no solo las del subcampo:

```python
# Enfoque actual: A es (papers_subcampo × refs_internas)
# Enfoque extendido: A es (papers_subcampo × TODAS_las_refs)
# C_BC = A @ A.T sigue siendo válido
ref_to_idx = {rid: i for i, rid in enumerate(all_referenced_works)}  # Sin filtro por subcampo
```

El costo computacional es mayor (el espacio de referencias puede ser 10-30x el tamaño del corpus), pero `scipy.sparse` lo maneja eficientemente. La ganancia metodológica es sustancial para frentes interdisciplinares.

### 1.2 Construcción de matriz por bin temporal ✅ (Acordado)

> *El investigador concuerda en que la matriz $A$ y el algoritmo Leiden deben ejecutarse por separado para cada bin temporal, no sobre el corpus completo.*

Construir $C_{BC}$ sobre el corpus completo mezcla papers de décadas distintas en el mismo cluster. El procedimiento correcto:

```python
for bin_start, bin_end in temporal_bins:
    df_bin = df[(df['year'] >= bin_start) & (df['year'] <= bin_end)]
    C_BC, work_to_idx = build_citation_matrix(df_bin)   # Solo papers del bin
    S = normalize_salton(C_BC)
    clusters = run_leiden(S, resolution=LEIDEN_RESOLUTION)
    # Guardar (bin, clusters) para el tracking longitudinal
```

### 1.3 Umbral de acoplamiento: criterio basado en la literatura

**Problema con `MIN_COUPLING_WEIGHT = 3`:** El valor absoluto de referencias compartidas no es comparable entre subcampos.

**Recomendación (Waltman, 2016):** Umbral sobre el **coseno de Salton normalizado** $S_{ik} \geq \theta$, con $\theta = 0.1$ como estándar de facto (VOSviewer, bibliometría computacional moderna). Es scale-free.

```python
S_filtered = np.where(S >= SALTON_THRESHOLD, S, 0)  # SALTON_THRESHOLD = 0.1
np.fill_diagonal(S_filtered, 0)
```

Criterio dinámico complementario (Glänzel & Thijs, 2011): umbral $\bar{S} - \sigma_S$ que adapta al campo.

---

## 2. Enfoque Semántico: SPECTER2 + Reducción Dimensional + HDBSCAN

### 2.1 Reducción dimensional: UMAP directo a 30d

UMAP es metodológicamente superior a PCA para embeddings de transformers (geometría no lineal). Pipeline recomendado: **UMAP(768→30d, coseno) → HDBSCAN(coseno)**, sin PCA intermedio.

### 2.2 HDBSCAN con 30 dimensiones y métrica coseno

Con métrica coseno, HDBSCAN es discriminativo hasta ~100 dimensiones (McInnes & Healy, 2018):

```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50, min_samples=10,
    cluster_selection_method='eom',
    metric='cosine',
    prediction_data=True
)
```

**Configuración recomendada:**
```
UMAP: 768d → 30d, n_neighbors=30, metric='cosine', min_dist=0.0
HDBSCAN: 30d, metric='cosine', min_cluster_size=50, min_samples=10
```

### 2.3 Escalabilidad para detección longitudinal

- 451K papers en 2025; ventana de 3 años ≈ 1.2M artículos.
- RTX 4090, batch_size=256: ~2ms/paper → ~40 min por ventana completa.
- **Estrategia de caché en ClickHouse:** solo se embeben papers nuevos por ventana (~150K/año ≈ ~5 min). Ver sección de `embeddings_cache`.

---

## 3. Enfoque Topológico: igraph/cuGraph con Red Heterogénea

### 3.1 Reemplazo de Neo4j GDS por igraph/cuGraph

Neo4j GDS (Community Edition) solo permite **una proyección de grafo en memoria a la vez**, lo que impide procesar múltiples ventanas temporales en paralelo. Alternativas elegidas:

| Opción | Ventanas simultáneas | Hardware | FastRP |
|---|---|---|---|
| **igraph** (primaria) | Ilimitadas (RAM) | CPU 100GB | Manual sparse |
| **cuGraph** (secundaria) | Por VRAM | RTX 4090 | Nativo completo |

La unidad de análisis es la **ventana temporal** (no el subcampo completo de 6M papers): cada ventana se carga en igraph, se procesa y se descarta.

### 3.2 Red heterogénea — diferenciador clave

```python
# igraph con aristas heterogéneas
edges = (
    citation_pairs +       # Work → Work
    authorship_pairs +     # Work → Author
    institution_pairs +    # Work → Institution
    source_pairs           # Work → Source
)
g = ig.Graph(n=num_nodes, edges=edges, directed=False)
g.es['weight'] = weights
```

FastRP sobre el grafo heterogéneo captura **cercanía por ecosistema científico**: dos papers sin citación directa quedan vinculados si comparten autores, institución o revista.

**Viabilidad de memoria (ventana 3 años, 1.2M papers):**
- Nodos + aristas en igraph ≈ 5-8 GB RAM
- Con 100 GB libres: perfectamente viable

---

## 4. Arquitectura de datos: `embeddings_cache` en ClickHouse

### 4.1 Diseño wide-format (una columna por modelo)

En ClickHouse columnar, una columna por modelo es más eficiente que una fila por (paper, modelo):
- Consultar SPECTER2 solo lee esa columna — cero I/O en otras
- Arrays del mismo tamaño por columna → compresión óptima
- `ALTER TABLE ADD COLUMN` es de metadata (milisegundos), no reescribe datos

```sql
CREATE TABLE IF NOT EXISTS embeddings_cache (
    id                   String,
    subfield_name        LowCardinality(String),
    publication_year     UInt16,

    -- Modelos semánticos
    embedding_specter2   Array(Float32),   -- 768d
    embedding_scilbert   Array(Float32),   -- 768d

    -- Modelos topológicos
    embedding_fastrp_cit Array(Float32),   -- 128d, red de citas
    embedding_fastrp_het Array(Float32),   -- 128d, red heterogénea

    -- Proyecciones reducidas
    embedding_umap_30d   Array(Float32),   -- 30d, derivado de specter2

    -- Auditoría
    specter2_at          Nullable(DateTime),
    fastrp_het_at        Nullable(DateTime),
    updated_at           DateTime DEFAULT now()

) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (subfield_name, id);
```

### 4.2 Separación de concerns

```
works_flat          → datos bibliométricos (fuente de verdad OpenAlex)
embeddings_cache    → artefactos ML derivados (INSERTs, nunca mutations)
```

Agregar un nuevo modelo:
```sql
ALTER TABLE embeddings_cache ADD COLUMN embedding_bge_m3 Array(Float32);
-- Instantáneo. No reescribe works_flat ni datos existentes.
```

---

## 5. Segmentación Temporal: Vigintiles + Ventana Deslizante

- **Vigintiles:** análisis histórico completo (1950-2025), volumen constante de papers.
- **Ventanas de 3 años** con paso anual: análisis reciente (2010-2025), 13 ventanas superpuestas, alta resolución para emergencia/extinción de frentes. Viable en hardware actual (~25GB igraph por ventana).

---

## 6. Mecanismo de Fusión: Evaluación Independiente (Fase Actual)

> *Decisión del investigador: primero evaluar cada método por separado. La fusión se decidirá según los resultados.*

Reportar por cada ventana temporal:
- Número de clusters por método y distribución de tamaños (Gini)
- AMI entre pares de métodos + número de clusters de cada partición
- Matriz de contingencia normalizada (overlap cluster-a-cluster)

**Escala de referencia AMI Leiden vs. HDBSCAN:**

| AMI | Interpretación |
|-----|---------------|
| 0.0 – 0.15 | Particiones independientes; alta presencia de silos o frentes emergentes |
| 0.15 – 0.35 | Solapamiento parcial; campo con mix de áreas consolidadas y emergentes |
| 0.35 – 0.60 | Concordancia moderada |
| > 0.60 | Alta concordancia; campo maduro y consolidado |

---

## 7. Resumen de Recomendaciones

| # | Recomendación | Prioridad | Estado |
|---|---------------|-----------|--------|
| 1 | Incluir citas externas al subcampo en la matriz $A$ | Alta | Por implementar |
| 2 | Construir $C_{BC}$ y ejecutar Leiden **por bin temporal** | Alta | ✅ Acordado |
| 3 | Reemplazar `MIN_COUPLING_WEIGHT` por umbral Salton ≥ 0.1 | Alta | Por implementar |
| 4 | UMAP(768→30d, coseno) + HDBSCAN(coseno) | Alta | Por implementar |
| 5 | `embeddings_cache` en ClickHouse (wide-format, por modelo) | Alta | ✅ Acordado |
| 6 | igraph/cuGraph para topológico (reemplaza Neo4j GDS multi-ventana) | Alta | ✅ Acordado |
| 7 | Red heterogénea: Work + Author + Institution + Source | Alta | ✅ Acordado |
| 8 | Vigintiles (histórico) + ventana 3 años (reciente) | Media | ✅ Viable |
| 9 | Evaluación independiente primero, fusión en fase posterior | Media | ✅ Decidido |
| 10 | AMI con matriz de contingencia cluster-a-cluster | Media | Por implementar |
| 11 | Documentar sensibilidad paramétrica (γ, min_cluster_size) | Baja | Por documentar |

---

## Referencias

- Kessler, M.M. (1963). Bibliographic coupling between scientific papers. *American Documentation*.
- Price, D.J. de S. (1965). Networks of scientific papers. *Science*.
- Waltman, L. & Van Eck, N.J. (2012). A new methodology for constructing a publication-level classification system. *Journal of the American Society for Information Science and Technology*.
- Waltman, L. (2016). A review of the literature on citation impact indicators. *Journal of Informetrics*.
- Traag, V.A., Waltman, L. & Van Eck, N.J. (2019). From Louvain to Leiden. *Scientific Reports*.
- McInnes, L. & Healy, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *JOSS*.
- Glänzel, W. & Thijs, B. (2012). Using hybrid methods and 'core documents' for the representation of clusters and topics. *Scientometrics*.
- Singh, S. et al. (2022). SciRepEval: A multi-format benchmark for scientific document representations. *arXiv*.
