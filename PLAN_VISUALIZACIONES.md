# Plan de Implementación: Visualizaciones Bibliométricas Avanzadas en Topics

Este documento detalla el plan de implementación para incorporar al proyecto **Topics** las visualizaciones y análisis avanzados inspirados en el generador de reportes de **RAGs**, adaptados para explotar de forma óptima los cachés precalculados a nivel de regiones, países e instituciones.

---

## 1. Visualizaciones a Incorporar y su Adaptación a Topics

El generador de reportes en RAGs utiliza varias visualizaciones sofisticadas que enriquecen sustancialmente el diagnóstico. Dado que el proyecto **Topics** cuenta con datasets agregados muy robustos en `data/cache_temas/`, estas visualizaciones se adaptarán de la siguiente manera:

### 1.1 Mapa Coroplético de Colaboración por País (Choropleth Map)
*   **Origen en RAGs:** Mapea los países colaboradores a partir de la lista de países por artículo.
*   **Adaptación en Topics:** Contamos con `_collab.parquet`, el cual registra las coautorías entre pares de países (ej: `country_a`, `country_b`, `count`).
    *   **Lógica:** Al seleccionar un país (ej. México / `MX`), filtraremos `_collab.parquet` para obtener la suma de publicaciones en coautoría con cada país socio.
    *   **Visualización:** Un gráfico `px.choropleth` interactivo que ilumine el mapa mundial en escala de azules según el volumen de coautorías del país objetivo.
    *   **Ubicación:** Pestaña "🤝 Colaboración" o como sección principal en el dashboard.

### 1.2 Red Topológica de Coautorías Internacionales (PyVis Graph)
*   **Origen en RAGs:** Red interactiva basada en Neo4j que conecta investigadores, artículos e instituciones.
*   **Adaptación en Topics:** Como no tenemos un motor de grafos Neo4j activo en este proyecto, usaremos los pares de coautoría de `_collab.parquet` para graficar la **estructura relacional de la ciencia** en el subcampo.
    *   **Lógica:** Usaremos la librería `pyvis.network` en Python. Los **nodos** serán los países y las **aristas** representarán las coautorías, con el grosor proporcional al volumen de trabajos conjuntos. Los nodos se colorearán de acuerdo con su región mundial (de `src/regions.py`).
    *   **Interactividad:** Grafo interactivo con físicas de repulsión (ForceAtlas2) incrustado mediante un iframe dentro de Streamlit o el reporte.

### 1.3 Gráfico de Dona y Áreas Apiladas de Acceso Abierto (Open Access Evolution)
*   **Origen en RAGs:** Gráfico de dona (Pie) de OA global e histórico.
*   **Adaptación en Topics:** Contamos con las columnas precalculadas `pct_oa_gold`, `pct_oa_diamond`, `pct_oa_green`, `pct_oa_hybrid`, `pct_oa_bronze`, `pct_oa_closed`.
    *   **Visualización 1 (Dona):** Un gráfico de dona con colores estandarizados (Dorado para Gold, Verde para Green, Azul para Hybrid, Gris para Closed, etc.) representando la distribución acumulada del periodo.
    *   **Visualización 2 (Áreas Apiladas):** Un gráfico `px.area` de evolución temporal que muestre la transición de la ciencia cerrada a la ciencia abierta a lo largo de los años para la entidad bajo estudio.

### 1.4 Cuadrante de Posicionamiento Geopolítico (Bubble Chart)
*   **Origen en RAGs:** Gráficos UMAP de investigadores.
*   **Adaptación en Topics:** Un scatter plot de cuadrantes cienciométricos para las regiones o países.
    *   **Ejes:** X: Volumen de producción en el subcampo (escala logarítmica); Y: Impacto de citación (FWCI promedio); Tamaño: % Top 10% (Excelencia).
    *   **Utilidad:** Permite ubicar geográficamente qué regiones/países tienen alta producción e impacto (Líderes), alta producción y bajo impacto (Productores masivos), baja producción y alto impacto (Nichos de excelencia) o baja producción y bajo impacto (Emergentes).

### 1.5 Visualización de Composición Taxonómica (Sunburst Chart)
*   **Origen en RAGs:** Sunburst de concentración taxonómica por dominio, campo, subcampo y tópico.
*   **Adaptación en Topics:** Contamos con la jerarquía en `df_hier` (Domain > Field > Subfield) y el desglose de producción por tópicos de la entidad.
    *   **Visualización:** Un gráfico `px.sunburst` interactivo que represente jerárquicamente cómo se distribuye la producción del subcampo en sus tópicos internos.
    *   **Interactividad:** Permite hacer clics para realizar un zoom dinámico e identificar áreas específicas de especialización en comparación con la media global.

### 1.6 Análisis de Alineación y Contribución con ODS (SDG Visualization)
*   **Origen en RAGs:** Matriz y narrativa de contribución al Desarrollo Sostenible de la ONU.
*   **Adaptación en Topics:** Explotaremos la métrica `sdg_docs` (publicaciones alineadas con ODS) precalculada en `_inst.parquet`.
    *   **Visualización 1 (Distribución ODS):** Gráfico de barras horizontales mostrando el volumen y porcentaje de artículos alineados con ODS por institución líder.
    *   **Visualización 2 (Matriz ODS):** Si se expone la granularidad de los ODS (del 1 al 17) desde ClickHouse, se generará un Heatmap interactivo que cruce instituciones/países con los ODS específicos relevantes del tema.

---

## 2. Cambios Propuestos en los Componentes

Para implementar estas mejoras de forma limpia y mantenible, se modificarán los siguientes archivos del proyecto:

---

### [Componente: Visualizaciones y Dashboard]

#### [MODIFY] [dashboard_tema.py](file:///c:/Users/jlja/Documents/Proyectos/Topics/dashboard_tema.py)
*   Integrar las funciones para renderizar el Mapa Coroplético, el Grafo PyVis, el Sunburst y la visualización de ODS.
*   Añadir el gráfico de dona de Acceso Abierto en la sección de detalles de la entidad.
*   Añadir una sección de "Geopolítica del Subcampo" con el cuadrante y el gráfico Sunburst.

#### [NEW] [viz_bibliometrics.py](file:///c:/Users/jlja/Documents/Proyectos/Topics/src/viz_bibliometrics.py)
*   Crear un nuevo módulo de visualización en `src/` que contenga funciones desacopladas para mantener limpio el código de Streamlit:
    *   `render_collaboration_map(df_collab, target_country)`: Retorna la figura coroplética.
    *   `render_pyvis_network(df_collab)`: Construye y exporta a HTML el grafo de coautoría internacional.
    *   `render_oa_donut(metrics_dict)`: Retorna el gráfico de dona de acceso abierto.
    *   `render_geopolitical_quadrants(df_data, period)`: Retorna el scatter plot de cuadrantes.
    *   `render_sunburst_hierarchy(df_hier, df_data)`: Genera el Sol Radiante taxonómico.
    *   `render_sdg_contributions(df_inst)`: Genera gráficos del impacto social e institucional (ODS).

---

## 3. Plan de Verificación

### Pruebas Manuales
1.  **Carga de Datos:** Validar que al seleccionar un subcampo (ej. *Pulmonary and Respiratory Medicine*), los cachés Parquet se carguen correctamente y alimenten las nuevas funciones de visualización.
2.  **Interactividad del Grafo:** Verificar que el grafo de PyVis se renderice correctamente en Streamlit dentro del iframe y que los nodos respondan al arrastre y zoom.
3.  **Filtrado por País:** Comprobar que al cambiar el país objetivo (de México a Brasil, por ejemplo), el mapa coroplético actualice dinámicamente las alianzas de coautoría de ese país específico.
4.  **Consistencia de Colores en OA:** Confirmar que los colores asignados a las vías de Acceso Abierto coincidan entre el gráfico de dona y la leyenda estándar de ciencia abierta.
