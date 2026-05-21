# Propuesta de Diseño: Generador de Reportes Cientométricos y Bibliométricos (Deep Research)
## Contexto del Proyecto: Topics (OpenAlex Analytical Pipeline)

Este documento detalla la conceptualización, estructura y diseño técnico para implementar un **Generador de Reportes Bibliométricos y Cientométricos Avanzados** en el proyecto **Topics**. 

El objetivo es transformar la rica data precalculada y almacenada en nuestros cachés regionales (`data/cache_temas/`) en informes ejecutivos auto-contenidos, visualmente espectaculares y con un análisis crítico profundo generado por Inteligencia Artificial (LLM).

---

## 1. Filosofía del Reporte: "Deep Research" y Multiescala

Un estudio cienciométrico riguroso no debe limitarse a un tablero interactivo (dashboard). Requiere de una **narrativa estructurada** que contextualice los datos y explique el comportamiento científico en múltiples niveles jerárquicos:

*   **Macro (Global):** ¿Cómo está evolucionando el subcampo a nivel mundial? ¿Cuál es la tasa de crecimiento del conocimiento en esta área?
*   **Meso (Regiones):** La geopolítica de la ciencia. Comparación de bloques globales (China vs. Norteamérica vs. Latinoamérica, etc.).
*   **Micro (Países e Instituciones):** El posicionamiento específico de países (con énfasis estratégico en México) y sus principales universidades y centros de investigación.

El reporte se estructurará con un **estilo editorial académico (Journal Style)**, inspirado en las publicaciones de revistas como *Nature*, *Scientometrics* o *Journal of Informetrics*: tipografía serif para lectura larga, diseño de doble columna para la narrativa, tarjetas KPI minimalistas y gráficos interactivos de alta fidelidad integrados perfectamente en un documento HTML autocontenido y apto para exportación limpia a PDF.

---

## 2. Estructura Propuesta del Reporte (Sección por Sección)

### Sección 1: Resumen Ejecutivo y Diagnóstico Global del Subcampo
*   **Propósito:** Ofrecer un diagnóstico inmediato del volumen, madurez y dinamismo del subcampo a nivel global.
*   **Indicadores Clave (KPI Cards):**
    *   Volumen histórico de documentos vs. producción reciente (últimos 5 años).
    *   Tasa de Crecimiento Anual Compuesto (CAGR).
    *   Impacto de Citación Normalizado por Categoría (FWCI) global promedio.
    *   Porcentaje de artículos de excelencia (% Top 10% y % Top 1% de citación mundial).
    *   Porcentaje de Acceso Abierto global.
*   **Análisis LLM:** Síntesis ejecutiva de la trayectoria histórica del tema, detectando si se encuentra en fase de consolidación, rápida expansión o declive.

### Sección 2: Dinámica Geopolítica y Participación Mundial (Share %)
*   **Propósito:** Analizar la distribución de la producción científica entre los 8 grandes bloques regionales del Norte y Sur Global y China.
*   **Componentes Visuales:**
    *   **Gráfica de Líneas de Participación (Share % Temporal):** Muestra el porcentaje de la producción mundial que representa cada región año con año (visibilizando fenómenos como el ascenso científico de China).
    *   **Cuadrante de Desempeño (Scatter Plot de Burbujas):** X: Volumen de producción en el periodo reciente (logarítmico); Y: FWCI promedio del periodo; Tamaño: Número de artículos en revistas Top 10% de citación.
*   **Análisis LLM:** Interpretación cienciométrica sobre el balance de poder científico mundial y la brecha en el impacto citacional entre el Norte y el Sur Global.

### Sección 3: Benchmarking y Posicionamiento de Países (Foco en México y LATAM)
*   **Propósito:** Detallar cómo se posicionan los países individuales en el subcampo, evaluando la contribución de México y otros países de Latinoamérica.
*   **Componentes Visuales:**
    *   **Tabla de Clasificación de Países Líderes:** Top 15 países ordenados por volumen, mostrando su FWCI, % Top 10%, % Open Access e Idioma dominante.
    *   **Red Topológica de Colaboración Internacional:** Grafo interactivo (utilizando *PyVis*) que mapea los lazos de coautoría entre países (de `_collab.parquet`). El grosor de los enlaces representará el volumen de coautorías, y los colores de los nodos agruparán las regiones.
*   **Análisis LLM:** Evaluación del nivel de internacionalización de la investigación del país objetivo y cómo la colaboración transfronteriza impacta la visibilidad y el FWCI promedio.

### Sección 4: Estructura e Identidad Temática del Subcampo
*   **Propósito:** Abrir la "caja negra" del subcampo para identificar las sub-temáticas o tópicos específicos y cómo se distribuye el interés científico.
*   **Componentes Visuales:**
    *   **Distribución Taxonómica (Sunburst o Treemap):** Gráfico que muestra la composición interna del subcampo desglosada por tópicos individuales y su peso volumétrico.
    *   **Trayectorias de Tópicos Emergentes (Line Chart):** Evolución temporal de los 10 tópicos principales en volumen o FWCI para detectar cuáles están ganando tracción reciente.
    *   **Métrica Especial:** Índice de Gini de concentración temática aplicado a las regiones y países seleccionados (para saber si un país diversifica su esfuerzo en todo el subcampo o se especializa en un nicho).
*   **Análisis LLM:** Explicación del perfil de especialización temática de México frente a los líderes globales y la media mundial.

### Sección 5: Paisaje Institucional y Estructura por Sectores
*   **Propósito:** Identificar quiénes son los actores de investigación clave (universidades, institutos de salud, empresas, agencias gubernamentales).
*   **Componentes Visuales:**
    *   **Gráfica de Áreas de Sectores (Evolución Sectorial):** Evolución del volumen de publicaciones clasificado por sector de las instituciones (Higher Education, Government, Healthcare, Corporate, Non-profit) a partir de `_inst_types.parquet`.
    *   **Burbuja de Benchmarking Institucional (Top 30 de la Región):** X: Documentos publicados; Y: FWCI promedio; Tamaño: Citas acumuladas; Color: País de origen.
    *   **Ranking Multi-Segmento:** Tablas separadas para las instituciones líderes a nivel Global, Regional (LATAM) y de México (utilizando la riqueza de los segmentos precalculados en `_inst.parquet`).
*   **Análisis LLM:** Diagnóstico de la madurez del ecosistema institucional. Por ejemplo, analizando el peso de las universidades públicas frente a los institutos de salud y el sector corporativo privado.

### Sección 6: Vías de Difusión Científica y Ciencia Abierta (Publishing & Open Science)
*   **Propósito:** Estudiar los hábitos editoriales del subcampo, identificando las revistas de preferencia y la adopción del modelo de Acceso Abierto.
*   **Componentes Visuales:**
    *   **Evolución Temporal del Acceso Abierto (Stacked Area Chart):** Evolución de las vías de publicación (Gold, Diamond, Green, Hybrid, Bronze, Closed) acumuladas a lo largo de los años.
    *   **Tabla de Canales de Publicación Top 15:** Revistas más utilizadas en el subcampo (de `_journals.parquet`), mostrando volumen de artículos recientes y su enlace directo a OpenAlex.
*   **Análisis LLM:** Discusión crítica sobre la inversión estimada en cargos por procesamiento de artículos (APC) frente al impacto real obtenido, y el papel de las revistas regionales de acceso abierto no comercial (vía Diamante y Verde) en el Sur Global.

### Sección 7: Alineación con Objetivos de Desarrollo Sostenible (ODS)
*   **Propósito:** Evaluar el impacto social y la aplicabilidad de la ciencia producida al alinearse con la agenda global de desarrollo de la ONU.
*   **Componentes Visuales:**
    *   **Matriz de Alineación ODS:** Tabla/Heatmap que cruza las regiones o países seleccionados con los 17 Objetivos de Desarrollo Sostenible (ODS) detectados algorítmicamente en sus publicaciones.
*   **Análisis LLM:** Discusión sobre cómo la investigación en este subcampo aborda desafíos urgentes (por ejemplo, ODS 3: Salud y Bienestar en medicina, u ODS 13: Acción por el Clima).

### Sección 8: Frentes de Investigación (Research Fronts) y Evolución Estructural (Futura Integración)
*   **Propósito:** Estudiar cómo ha evolucionado cognitivamente la estructura del subcampo a través del tiempo, detectando el nacimiento y evolución de frentes específicos de investigación.
*   **Componentes Visuales:**
    *   **Grafo Longitudinal de Frentes:** Grafo que conecta los clusters de co-citación a lo largo de diferentes periodos temporales (mostrando cómo un frente se divide, se fusiona o emerge).
    *   **Visualización en Espacio Semántico (SOM / UMAP):** Mapa que muestra la cercanía temática y la densidad actual de los frentes de investigación activos.
*   **Análisis LLM:** Detección de los frentes calientes ("hot fronts") en la frontera del conocimiento y qué tecnologías o teorías revolucionarias los están impulsando.

---

## 3. Arquitectura Técnica y Flujo de Implementación

Para asegurar robustez, velocidad y coherencia de diseño, la arquitectura del generador de reportes se apoyará en los siguientes pilares del proyecto actual:

```mermaid
graph TD
    A[CLI / Dashboard] -->|Especifica Subcampo e Indicadores| B(report_generator.py)
    B -->|Carga de datos eficientes| C[pipeline_topic/data_processor.py]
    C -->|Lectura de Parquets| D[data/cache_temas/*.parquet]
    B -->|Llamadas al LLM Local| E[lib/llm_utils.py]
    E -->|API Request con Baja Temp 0.2| F[LM Studio / Remote LLM]
    B -->|Renderizado de Figuras| G[Plotly & PyVis]
    B -->|Ensamblador HTML/CSS| H[report_{subcampo}.html]
```

### 3.1 Carga de Datos Optimizada
En lugar de consultar la base de datos OLAP (ClickHouse) directamente al generar el reporte, el generador usará exclusivamente las funciones de carga en `pipeline_topic/data_processor.py` (`load_subfield_data`, `load_collaboration_data`, `load_institutional_data`, etc.). Esto garantiza:
1.  **Velocidad Instantánea:** La lectura de archivos Parquet locales toma milisegundos.
2.  **Aislamiento:** El generador de reportes puede funcionar de forma 100% offline si los cachés ya han sido calculados en el dashboard.

### 3.2 Generación Narrative con LLM (Zero-Hallucination)
Reutilizaremos la infraestructura de `lib/llm_utils.py` para construir un cliente HTTP robusto conectado a LM Studio o al LLM remoto.
*   **Prompts Estrictos:** Cada prompt enviado al LLM recibirá las tablas de datos agregadas en formato Markdown o JSON plano. Se le instruirá explícitamente al modelo actuar como un "analista cienciométrico sénior de una prestigiosa academia de ciencias", escribiendo de forma sobria, académica, sin adjetivos sensacionalistas ni hipérboles.
*   **Control de Temperatura:** Temperatura fijada en `0.2` para asegurar que el modelo se ciña estrictamente a los números proporcionados y no "invente" métricas o tendencias inexistentes.

### 3.3 Motor Visual y Estilos (Journal Style)
*   **Plotly Offline:** Todos los gráficos interactivos se generarán con Plotly en Python y se convertirán a cadenas HTML utilizando `fig.to_html(include_plotlyjs='cdn', full_html=False)`.
*   **PyVis para Grafos:** La red de coautoría de países se creará con PyVis, configurando la física óptima de repulsión de nodos, y se incrustará usando un `iframe srcdoc` seguro dentro del reporte HTML.
*   **Estilo Editorial CSS:**
    *   **Contenedor:** Ancho máximo de `1100px`, centrado, con fondo de página ligeramente gris cálido (`#fafafa`) y fondo del reporte en blanco puro (`#ffffff`) con sutiles sombras tipográficas para emular papel grueso de imprenta.
    *   **Tipografía:** Google Fonts - `Merriweather` para el cuerpo del texto (legibilidad óptima serif) y `Outfit` o `Open Sans` para encabezados y etiquetas de datos (sans-serif moderno).
    *   **Layout de Texto:** Uso de columnas múltiples de CSS (`column-count: 2`) para la narrativa del análisis cienciométrico, logrando la maquetación clásica de un artículo científico.
    *   **Letra Capital (Dropcap):** Implementación de letra capital gigante en el inicio del Resumen Ejecutivo para un acabado editorial premium.
    *   **Adaptación de Impresión (Print-CSS):** Reglas específicas `@media print` para forzar saltos de página inteligentes (`page-break-inside: avoid`) en gráficos y tablas, eliminando fondos innecesarios para permitir una exportación limpia a PDF de tamaño carta/A4.

---

## 4. Plan de Acción para la Implementación

1.  **Creación del Módulo Base (`Report/report_generator.py`):** Estructura del script ejecutable mediante consola y CLI (`python -m Report.report_generator --subfield "Pulmonary and Respiratory Medicine" --entity "México"`).
2.  **Desarrollo del Ensamblador HTML y Hoja de Estilos:** Crear el layout base con las tipografías Merriweather/Outfit y el Grid de diseño.
3.  **Implementación de Sub-Reportes Gráficos (Plotly + PyVis):** Escribir las funciones de Python para mapear los cachés a gráficos Plotly (Share, Cuadrante regional, Red de Países, Sectores, Canales de Publicación, ODS, etc.).
4.  **Integración y Refinamiento de Prompts del LLM:** Diseñar las funciones que alimentan al LLM local con los datos consolidados para redactar las interpretaciones académicas de cada sección.
5.  **Pruebas de Exportación y Optimización Print-CSS:** Verificar el peso del archivo HTML autocontenido (debe rondar menos de 2MB usando Plotly CDN) y validar que la exportación nativa a PDF desde el navegador (Ctrl+P) rinda un documento impecable.
6.  **Integración en la UI del Dashboard Streamlit:** Añadir un botón en la barra lateral del dashboard para que el usuario pueda lanzar la generación del reporte en tiempo real para cualquier subcampo seleccionado y descargarlo inmediatamente.
