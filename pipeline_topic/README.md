# Dashboard Temático Global (OpenAlex)

## Descripción General

`dashboard_tema.py` es una aplicación analítica integral desarrollada en Streamlit que proporciona inteligencia bibliométrica de alto desempeño estructurada por subcampos científicos (Subfields) y tópicos (Topics). El proyecto consolida métricas de impacto, colaboración, tipología de acceso abierto e influencia institucional basándose en el gran corpus de datos abiertos de OpenAlex.

## Arquitectura de Datos (`pipeline_topic/`)

Para asegurar de que la aplicación corra de manera fluida y fluida procesando millones de artículos, la arquitectura se divide en tres niveles fundamentales para evitar cuellos de botella computacionales en Python:

1. **Motores de Agregación (ClickHouse):** 
   Se utiliza todo el poder del modelo de datos tipo columna de ClickHouse. El dashboard no lee directamente de la tabla `works`, sino que depende fuertemente de tablas materializadas tipo `SummingMergeTree` (típicamente `summing_subfield_metrics` y `summing_subfield_inst_metrics`). Estas tablas colapsan y pre-calculan las métricas necesarias por año, tópico, país y fuente.
   
2. **Capa de Lógica y Procesamiento (`compute_metrics.py`):**
   Actúa como el puente interactivo. Mediante el uso de consultas SQL secuenciales pre-diseñadas extrae datos base para tres niveles territoriales (Mundo, Región y País). A nivel de backend (en Python / Pandas) se realizan únicamente las transformaciones probabilísticas y cálculos precisos de porcentajes: Field-Weighted Citation Impact (FWCI), percentiles normalizados y demografía Open Access/Idiomática. Los DataFrames resultantes se "imprimen" de forma persistente en archivos `.parquet` (caché local).
   
3. **Capa UI y Renderizado Rápido (`dashboard_tema.py` & `data_processor.py`):** 
   La aplicación front-end sólo consume las memorias parquet (`data/cache_temas/`). Mapea y enruta los datos a gráficos interactivos en Plotly y KPIs premium sin latencia perceptible. Si los datos no están oxigenados, solicita expresamente al backend su recálculo de manera dinámica tras validación del usuario (el botón "Forzar Recálculo").

## Módulos de la Aplicación

- **Evolución y Benchmarking:** Vistas temporales sincronizadas para hacer benchmarking inmediato de Producción, Calidad (% Top 1%, Top 10%) e Impacto entre cualquier ecosistema de países definidos.
- **Liderazgo Institucional:** Nuevo módulo capaz de visualizar y descargar las instituciones élite del mundo en formato _Scatter_ a partir de su volumen, impacto global e intersección en ODS (Objetivos de Desarrollo Sostenible).
- **Cartografía de Publicación:** Ranking automatizado de las revistas predilectas globales en un nicho de conocimiento específico.
- **Redes y Alianzas:** Algoritmos dedicados para visualizar los canales de Colaboración Internacional en los subcampos selectos.

## Requisitos de Ejecución

Para ejecutar o desarrollar sobre esta suite, es imperativo:
- Instancia activa de `ClickHouse`.
- Conectividad a la misma base y variables de entorno declaradas correctamente (`CH_HOST`, `CH_USER`, `CH_PASSWORD`, `CH_DATABASE` etc.) en `.env`.

**Lanzamiento:**
```bash
streamlit run dashboard_tema.py
```
