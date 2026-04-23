# Esquema de Pruebas (Sandbox): Frentes de Investigación

Este documento detalla el plan para validar el pipeline multimodal con un subconjunto controlado de datos antes de proceder a la implementación completa sobre ClickHouse.

---

## 1. Configuración del Entorno

### Instalación de Dependencias Faltantes
Ejecutar en la terminal del proyecto:
```bash
pip install igraph leidenalg hdbscan
```
*Nota: `igraph` debe instalarse antes que `leidenalg`.*

### Infraestructura Local
- **Neo4j:** Asegurar que el contenedor Docker esté corriendo.
  - URI sugerida: `bolt://localhost:7687` (o el puerto configurado).
  - Usuario/Password: `neo4j` / `password` (o el configurado en el archivo .env).
- **ClickHouse:** Conexión vía VPN activa para la extracción inicial.

---

## 2. Selección del Subconjunto de Datos (Sandbox)

Para que las pruebas sean ágiles, seleccionaremos un corpus de **~500 a 1000 papers**.

**Estrategia de Selección:**
Usar un término muy específico dentro de un subcampo para garantizar densidad de citas.
- **Subcampo:** `Pulmonary and Respiratory Medicine` (ID: `2737`)
- **Filtro:** Títulos que contengan "COVID-19" o "SARS-CoV-2" entre 2020 y 2021.

**Query de Extracción (Sandbox):**
```sql
SELECT 
    work_id, title, abstract, publication_year, referenced_works
FROM works
WHERE arrayExists(x -> x = '2737', subfields_ids)
  AND (hasToken(lower(title), 'covid') OR hasToken(lower(title), 'sars'))
  AND publication_year BETWEEN 2020 AND 2021
LIMIT 1000
```

---

## 3. Protocolo de Prueba "Mini-Pipeline"

Crearemos un script `fronts/sandbox_test.py` que realice los siguientes pasos:

### Paso A: Extracción y Local Storage
1. Ejecutar la query en ClickHouse.
2. Guardar el resultado en un archivo local `data/sandbox_papers.parquet`.

### Paso B: Carga en Neo4j Local
1. Limpiar el grafo de prueba en Neo4j Docker.
2. Cargar los papers y sus relaciones de citación (solo entre ellos para el sandbox).
3. Cargar relaciones de Autores e Instituciones (opcional para Fase 1, necesario para Fase 3).

### Paso C: Ejecución de Detección
1. **Estructural:** Generar matriz `C_BC` y ejecutar Leiden.
2. **Semántico:** Generar embeddings (solo 1000 nodos es rápido en la 4090) y ejecutar HDBSCAN.
3. **Topológico:** Ejecutar FastRP en Neo4j Docker y traer los clusters.

---

## 4. Criterios de Éxito (Validación)

Se considera que el sandbox es exitoso si:
1. **Conectividad:** Se logra extraer datos de ClickHouse y subirlos a Neo4j Docker sin errores de red.
2. **Matrices:** Se genera una matriz `C_BC` dispersa (sparse) válida.
3. **Coherencia:** Los clusters detectados por Leiden y HDBSCAN tienen un `AMI > 0.1` (indicando que hay estructura real, no ruido).
4. **Naming:** El LLM local genera nombres coherentes para al menos 2 clusters del sandbox.

---

## 5. Script de Inicialización (Sugerido)

```python
# fronts/sandbox_init.py
import pandas as pd
from fronts.clickhouse_queries import get_sandbox_data

def run_sandbox():
    print("📥 Cargando datos desde ClickHouse (VPN)...")
    df = get_sandbox_data(limit=1000)
    df.to_parquet("data/sandbox_papers.parquet")
    print(f"✅ Guardados {len(df)} papers para pruebas.")
    
    # Próximos pasos: Ingesta en Neo4j y validación de algoritmos
```
