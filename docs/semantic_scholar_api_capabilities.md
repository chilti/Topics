# Semantic Scholar API — Capacidades, Límites y Acceso a Contexto de Citas

**Fecha de investigación:** 2026-06-12  
**Contexto:** Evaluación comparativa con Scopus (`AbstractRetrieval`) para determinar si Semantic Scholar ofrece mejores límites de recuperación y, especialmente, acceso a la **intención/contexto de las citas** (Highly Influential, Background, Methods, Results).

---

## 1. ¿Qué es la Semantic Scholar Academic Graph API (S2AG)?

API REST gratuita de [Allen Institute for AI](https://allenai.org/), con acceso abierto a más de **200 millones de artículos científicos**. Ofrece:

- Recuperación de registros completos por DOI, S2 Paper ID, ArXiv ID, PubMed ID, etc.
- Citas con **clasificación por intención** y fragmentos de texto contextuales
- Embeddings vectoriales (SPECTER2) por defecto
- Descarga masiva del corpus completo (S2AG Datasets)
- **Sin muros de pago institucionales** — completamente gratuito

---

## 2. Rate Limits — Comparativa con Scopus

| Tier | Requests/segundo | Costo | Cómo obtener |
|---|---|---|---|
| **Sin API key** | ~20 req/min (pool compartido) | Gratis | Directo |
| **Con API key (estándar)** | **1 req/s (dedicado)** | Gratis | Registro en [S2 API Portal](https://www.semanticscholar.org/product/api) |
| **Con API key (académico/partner)** | **10+ req/s** (revisión) | Gratis | Solicitud justificada |
| **Datasets bulk download** | Sin límite (descarga directa) | Gratis | `api.semanticscholar.org/corpus` |

### Comparativa directa con Scopus

| Métrica | Scopus `AbstractRetrieval` | **Semantic Scholar** |
|---|---|---|
| Costo | Requiere suscripción inst. | **Gratuito** |
| Cuota semanal | ~10,000 req/semana | **Sin cuota semanal** |
| Velocidad estándar | 3–9 req/s | 1 req/s (ampliable) |
| Batch endpoint | ❌ No | ✅ Hasta 500 docs/req |
| Descarga masiva | ❌ No | ✅ Corpus completo |
| Contexto de citas | ❌ Solo `citedby_count` | ✅ **Intención + fragmento de texto** |
| Open Access | Solo metadato | ✅ PDF URL si disponible |
| Embeddings | ❌ No | ✅ SPECTER2 incluido |

### Documentos recuperables por período (con API key estándar, 1 req/s)

> Con el **endpoint batch** (`paper/batch`), cada request puede traer hasta **500 documentos**, cambiando radicalmente los números:

| Período | Sin batch (1 doc/req) | **Con batch (500 docs/req)** |
|---|---|---|
| Por minuto | 60 docs | **30,000 docs** |
| Por hora | 3,600 docs | **1,800,000 docs** |
| Por día | 86,400 docs | **~43 millones docs** |
| Por semana | ~600,000 docs | Sin límite práctico |

> **Nota:** El batch endpoint es la clave. En la práctica, con metadata completa y esperas por red, el throughput real es ~100K–500K docs/día.

---

## 3. Campos disponibles por documento

### 3.1 Endpoint de paper individual
```
GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=...
```

Identificadores soportados como `{paper_id}`:
- DOI: `10.1016/j.xxx.2020.xxx`
- S2 Paper ID: hash de 40 caracteres
- `ARXIV:2106.15928`
- `PMID:33456789`
- `DBLP:conf/acl/...`

### 3.2 Campos disponibles (todos opcionales, separados por coma)

| Campo | Descripción |
|---|---|
| `paperId` | ID interno de Semantic Scholar |
| `corpusId` | ID del corpus S2AG |
| `externalIds` | Dict con DOI, PubMed, ArXiv, DBLP, MAG, etc. |
| `url` | URL al perfil del paper en S2 |
| `title` | Título |
| `abstract` | Resumen completo |
| `venue` | Nombre de la revista/conferencia |
| `publicationVenue` | Objeto con ID, nombre, tipo, URL |
| `year` | Año de publicación |
| `publicationDate` | Fecha exacta (YYYY-MM-DD) |
| `publicationTypes` | Lista: `JournalArticle`, `Review`, `Conference`, etc. |
| `journal` | Objeto: `name`, `pages`, `volume` |
| `authors` | Lista con `authorId`, `name` |
| `citationCount` | Número total de citas |
| `referenceCount` | Número de referencias |
| `influentialCitationCount` | Conteo de citas altamente influyentes |
| `isOpenAccess` | Boolean |
| `openAccessPdf` | Objeto con `url` y `status` del PDF |
| `fieldsOfStudy` | Clasificación amplia (Computer Science, Medicine, etc.) |
| `s2FieldsOfStudy` | Clasificación S2 con `category` y `source` |
| `citations` | Lista de papers que citan a este |
| `references` | Lista de referencias del paper |
| `embedding` | Vector SPECTER2 (768 dimensiones) |
| `tldr` | Resumen IA de una oración (`text` + `model`) |
| `citationStyles` | Estilos de cita (APA, BibTeX, etc.) |

---

## 4. Contexto e Intención de Citas ⭐ (característica exclusiva)

Esta es la funcionalidad más poderosa frente a Scopus. S2 clasifica **cómo** y **por qué** un paper es citado, con acceso al texto exacto donde ocurre la cita.

### 4.1 Endpoint de citas con contexto
```
GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations
    ?fields=intents,isInfluential,contexts,citingPaper.title,citingPaper.year
    &limit=1000
```

### 4.2 Campos de cada cita en la respuesta

| Campo | Tipo | Descripción |
|---|---|---|
| `isInfluential` | `boolean` | Cita **Highly Influential** (ML model) |
| `intents` | `list[str]` | Lista de intenciones: `["background"]`, `["methodology"]`, `["result"]` |
| `contexts` | `list[str]` | **Fragmentos textuales** exactos del paper citante donde aparece la referencia |
| `citingPaper` | `object` | Metadata del paper que cita (título, año, autores, etc.) |

### 4.3 Las 4 categorías de la interfaz de Scopus/S2

Lo que ves en la interfaz de Semantic Scholar:

| Etiqueta en UI | Campo API | Descripción del modelo |
|---|---|---|
| **Highly Influential Citations** | `isInfluential = true` | El modelo ML determinó que la cita es fundamental para el paper citante (alto re-uso, uso en métodos o resultados, frecuencia de mención) |
| **Background Citations** | `intents = ["background"]` | El paper es citado para dar contexto histórico, justificar relevancia o dar información de fondo |
| **Methods Citations** | `intents = ["methodology"]` | El paper es citado como procedimiento establecido o técnica experimental |
| **Results Citations** | `intents = ["result"]` | El paper es citado al extender, comparar o contrastar resultados |

> **Limitación clave:** La clasificación de intención solo está disponible cuando S2 tiene acceso al **texto completo** del paper citante. Para papers sin acceso a full-text, `intents` y `contexts` pueden ser nulos/vacíos.

### 4.4 Ejemplo completo en Python

```python
import requests
import time

API_KEY = "TU_API_KEY"  # Obtener en semanticscholar.org/product/api
HEADERS = {"x-api-key": API_KEY}

def get_citations_with_context(doi: str, limit: int = 1000) -> list:
    """
    Recupera las citas de un paper con su intención, influencia y contexto textual.
    Soporta paginación automática.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}/citations"
    params = {
        "fields": "intents,isInfluential,contexts,citingPaper.title,citingPaper.year,citingPaper.authors,citingPaper.externalIds",
        "limit": min(limit, 1000),
    }
    
    all_citations = []
    offset = 0
    
    while True:
        params["offset"] = offset
        resp = requests.get(url, params=params, headers=HEADERS)
        
        if resp.status_code == 429:
            print("Rate limit alcanzado. Esperando 5s...")
            time.sleep(5)
            continue
        
        data = resp.json()
        batch = data.get("data", [])
        all_citations.extend(batch)
        
        # Verificar si hay más páginas
        if not data.get("next") or len(batch) == 0:
            break
        offset += len(batch)
        time.sleep(1)  # Respetar 1 req/s
    
    return all_citations


def classify_citations(citations: list) -> dict:
    """Clasifica las citas por tipo, igual que en la UI de Semantic Scholar."""
    return {
        "highly_influential": [c for c in citations if c.get("isInfluential")],
        "background":         [c for c in citations if "background" in c.get("intents", [])],
        "methodology":        [c for c in citations if "methodology" in c.get("intents", [])],
        "result":             [c for c in citations if "result" in c.get("intents", [])],
        "no_intent_info":     [c for c in citations if not c.get("intents")],
    }


# --- Uso ---
doi = "10.1016/j.softx.2019.100263"
citations = get_citations_with_context(doi)
classified = classify_citations(citations)

print(f"Total citas recuperadas: {len(citations)}")
print(f"Highly Influential:      {len(classified['highly_influential'])}")
print(f"Background:              {len(classified['background'])}")
print(f"Methods (Methodology):   {len(classified['methodology'])}")
print(f"Results:                 {len(classified['result'])}")
print(f"Sin clasificar:          {len(classified['no_intent_info'])}")

# Ver los fragmentos de texto de una cita influyente
if classified["highly_influential"]:
    c = classified["highly_influential"][0]
    print(f"\nEjemplo Highly Influential:")
    print(f"  Titulo citante : {c['citingPaper']['title']}")
    print(f"  Intenciones    : {c['intents']}")
    print(f"  Contextos      : {c['contexts']}")
```

### 4.5 Endpoint de recuperación en batch (para enriquecer ClickHouse)

```python
import requests

def batch_enrich_papers(dois: list, api_key: str = None) -> list:
    """
    Recupera metadata completa de hasta 500 papers en una sola llamada.
    Ideal para enriquecer registros de ClickHouse.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    
    fields = ",".join([
        "paperId", "externalIds", "title", "abstract",
        "year", "publicationDate", "publicationTypes",
        "journal", "venue", "authors",
        "citationCount", "influentialCitationCount", "referenceCount",
        "isOpenAccess", "openAccessPdf",
        "fieldsOfStudy", "s2FieldsOfStudy",
        "tldr", "embedding"   # SPECTER2 embedding incluido
    ])
    
    results = []
    for i in range(0, len(dois), 500):  # Chunks de 500
        chunk = dois[i:i+500]
        resp = requests.post(
            url,
            params={"fields": fields},
            json={"ids": chunk},
            headers=headers
        )
        if resp.status_code == 200:
            results.extend(resp.json())
        time.sleep(1)
    
    return results
```

---

## 5. Biblioteca Python oficial

```bash
pip install semanticscholar
```

```python
from semanticscholar import SemanticScholar

sch = SemanticScholar(api_key="TU_API_KEY")

# Paper por DOI
paper = sch.get_paper("10.1016/j.softx.2019.100263")
print(paper.title, paper.abstract, paper.influentialCitationCount)

# Citas con contexto
citations = sch.get_paper_citations(
    paper.paperId,
    fields=["intents", "isInfluential", "contexts", "citingPaper"]
)
```

---

## 6. Descarga masiva del corpus completo (S2AG Datasets)

Para proyectos que requieren millones de registros, S2 ofrece snapshots completos descargables:

```bash
# Via API de datasets
GET https://api.semanticscholar.org/datasets/v1/release/latest

# Datasets disponibles:
# - papers         : metadata completa de todos los papers
# - abstracts      : abstracts separados (archivo grande)
# - authors        : todos los autores
# - citations      : tabla completa de citas (incluyendo intents)
# - embeddings     : vectores SPECTER2 para todos los papers
# - tldrs          : resúmenes IA de todos los papers
```

> Tamaño aproximado: papers ~50GB comprimido, embeddings ~200GB.

---

## 7. Resumen ejecutivo — ¿Cuándo usar cada fuente?

| Necesidad | Scopus `AbstractRetrieval` | **Semantic Scholar** |
|---|---|---|
| Registro completo por DOI | ✅ | ✅ |
| Abstract | ✅ | ✅ |
| Afiliaciones institucionales detalladas | ✅ (mejor) | Parcial |
| Referencias completas | ✅ (view=FULL) | ✅ |
| **Contexto de citas (Background/Methods/Results)** | ❌ | ✅ **Exclusivo** |
| **Highly Influential flag** | ❌ | ✅ **Exclusivo** |
| **Fragmentos textuales de las citas** | ❌ | ✅ **Exclusivo** |
| Embeddings vectoriales | ❌ | ✅ (SPECTER2) |
| Resumen IA (TLDR) | ❌ | ✅ |
| Batch retrieval | ❌ | ✅ 500/req |
| Sin cuota semanal | ❌ (10K/semana) | ✅ |
| Cobertura de libros/tesis | Limitada | Amplia |
| Datos de financiamiento | ✅ | ❌ |
| EID / ASJC subject areas | ✅ | ❌ |

### Recomendación de estrategia híbrida

```
Scopus AbstractRetrieval → para: afiliaciones completas, EID, ASJC, financiamiento
Semantic Scholar        → para: contexto de citas, highly influential,
                                 embeddings, TLDR, enriquecimiento masivo
```

---

## 8. Cómo obtener API key

1. Ir a [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
2. Hacer clic en "Get Started" → registrarse con correo institucional
3. Llenar el formulario con descripción del proyecto de investigación
4. Recibes la key por correo en 1–3 días hábiles
5. Para límites mayores a 1 req/s: escribir a **s2-api@semanticscholar.org**

---

## 9. Referencias

- [Semantic Scholar API Documentation](https://api.semanticscholar.org/api-docs/)
- [S2AG Product Page](https://www.semanticscholar.org/product/api)
- [S2 Datasets Corpus](https://api.semanticscholar.org/corpus)
- [Python Library: semanticscholar](https://pypi.org/project/semanticscholar/)
- Paper del modelo de intención: *"Structural Scaffolds for Citation Intent Classification"* (Cohan et al., 2019)
- Reporte comparativo de Scopus: [`docs/scopus_api_abstract_retrieval_limits.md`](./scopus_api_abstract_retrieval_limits.md)
