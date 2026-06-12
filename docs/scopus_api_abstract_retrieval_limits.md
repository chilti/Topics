# Scopus API — AbstractRetrieval: Límites y Capacidad de Recuperación

**Fecha de investigación:** 2026-06-12  
**Contexto:** Evaluación de viabilidad para enriquecer registros de ClickHouse con metadata completa de Scopus (abstract, afiliaciones, referencias) usando la clase `AbstractRetrieval` de pybliometrics.

---

## ¿Qué es AbstractRetrieval?

A diferencia de `ScopusSearch` (búsqueda masiva, ~15 campos, límite de 5,000 resultados por query), `AbstractRetrieval` recupera el **registro completo de un documento individual** a partir de:

- **DOI**: `AbstractRetrieval("10.1016/j.xxx.2020.xxx", id_type="doi")`
- **EID**: `AbstractRetrieval("2-s2.0-85068792958", id_type="eid")`
- **Scopus ID**: `AbstractRetrieval("85068792958", id_type="scopus_id")`

---

## Comparativa: ScopusSearch vs AbstractRetrieval

| Característica | `ScopusSearch` | `AbstractRetrieval` |
|---|---|---|
| Uso principal | Búsqueda masiva | Documento individual |
| Límite de resultados | 5,000 por query | Sin límite (1 doc/llamada) |
| Campos disponibles | ~15 campos básicos | ~50+ campos completos |
| Abstract | ❌ No incluido | ✅ Completo |
| Referencias | ❌ No | ✅ Con `view="FULL"` |
| Afiliaciones completas | Parcial | ✅ Con IDs institucionales |
| Financiamiento | ❌ No | ✅ Incluido |
| Requiere suscripción inst. | No (limitado) | Para `view="FULL"` sí |

---

## Campos disponibles con `view="FULL"`

| Campo pybliometrics | Descripción |
|---|---|
| `ab.title` | Título completo |
| `ab.abstract` | Resumen completo |
| `ab.authors` | Lista con `auid`, `surname`, `given_name`, `affiliation` |
| `ab.affiliation` | Afiliaciones con institución, país, ciudad |
| `ab.references` | Lista de referencias citadas |
| `ab.authkeywords` | Keywords del autor |
| `ab.idxterms` | Términos indexados por Scopus |
| `ab.subject_areas` | Áreas temáticas ASJC |
| `ab.citedby_count` | Número de citas |
| `ab.doi` / `ab.eid` | Identificadores |
| `ab.coverDate` | Fecha de publicación |
| `ab.publicationName` | Nombre de la revista |
| `ab.issn` / `ab.eissn` | ISSN electrónico |
| `ab.volume` / `ab.issueIdentifier` | Volumen / número |
| `ab.openaccess` | Estado Open Access |
| `ab.fund_agency_name` | Agencias financiadoras |

---

## Límites de la API por tipo de cuenta

| Parámetro | Cuenta estándar | Cuenta institucional (UNAM) |
|---|---|---|
| **Cuota semanal** | ~5,000 req/semana | ~10,000 req/semana* |
| **Throttle (velocidad máx.)** | 3 req/s | 3–9 req/s |
| **Reset** | Cada 7 días (rolling) | Cada 7 días (rolling) |
| **`view="FULL"`** | ❌ Bloqueado | ✅ Disponible |

> *Los límites exactos dependen del acuerdo institucional. Se verifican en los headers de respuesta `X-RateLimit-*`.

---

## Proyección de documentos recuperables

Usando velocidad conservadora de **2 req/s** (para evitar errores 429):

| Período | Con 5,000/semana | Con 10,000/semana |
|---|---|---|
| **Por día** | ~714 docs | ~1,428 docs |
| **Por semana** | **5,000 docs** | **10,000 docs** |
| **Por mes (4 semanas)** | ~20,000 docs | ~40,000 docs |
| **Por año** | ~260,000 docs | ~520,000 docs |

> **Nota importante:** pybliometrics tiene caché local por defecto. Una vez descargado, un registro no vuelve a consumir cuota. Los números anteriores son de llamadas *nuevas* al API.

---

## Estrategia de enriquecimiento sugerida

Con ~200K papers en ClickHouse y una cuota de 10,000/semana:

- **Enriquecimiento completo**: ~5 meses corriendo continuamente.
- **Con priorización** (solo registros sin abstract o sin afiliaciones completas, estimado ~30K): **< 1 mes**.

### Orden de prioridad recomendado

1. Papers de investigadores SNII (ya en Neo4j) sin abstract
2. Papers con DOI pero sin EID de Scopus
3. Papers con alta citación sin afiliaciones completas
4. Resto del corpus

---

## Cómo verificar tus límites reales

```python
import requests

r = requests.get(
    "https://api.elsevier.com/content/abstract/doi/10.1016/j.softx.2019.100263",
    headers={
        "X-ELS-APIKey": "TU_API_KEY",
        "Accept": "application/json"
    }
)

print("Límite semanal:", r.headers.get("X-RateLimit-Limit"))
print("Restante:      ", r.headers.get("X-RateLimit-Remaining"))
print("Reset en:      ", r.headers.get("X-RateLimit-Reset"))  # Unix timestamp
```

---

## Uso básico con pybliometrics

```python
from pybliometrics.scopus import AbstractRetrieval
import time

def enrich_from_scopus(doi: str) -> dict:
    """Recupera el registro completo de un documento Scopus por DOI."""
    try:
        ab = AbstractRetrieval(doi, id_type="doi", view="FULL")
        return {
            "title":         ab.title,
            "abstract":      ab.abstract,
            "authors":       ab.authors,        # lista de namedtuples
            "affiliations":  ab.affiliation,    # lista de namedtuples
            "references":    ab.references,     # lista de namedtuples
            "keywords":      ab.authkeywords,
            "subject_areas": ab.subject_areas,
            "citations":     ab.citedby_count,
            "eid":           ab.eid,
            "openaccess":    ab.openaccess,
            "fund_agency":   ab.fund_agency_name,
        }
    except Exception as e:
        print(f"[!] Error para {doi}: {e}")
        return {}
    finally:
        time.sleep(0.5)  # ~2 req/s, conservador para respetar throttle
```

---

## Referencias

- [Elsevier Developer Portal — API Key Settings](https://dev.elsevier.com/api_key_settings.html)
- [pybliometrics documentation](https://pybliometrics.readthedocs.io/)
- [Elsevier Support — Rate Limits](https://elsevier.support)
- Implementación existente en el proyecto: [`ingestion/ingest_apis.py`](../ingestion/ingest_apis.py) (usa `AuthorRetrieval`)
- Parser CSV de Scopus: [`ingestion/scopus_csv_parser.py`](../ingestion/scopus_csv_parser.py)
