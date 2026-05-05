import json
import sys
from pathlib import Path
from typing import List

# Asegurar que el directorio raíz está en el path para importar lib.llm_utils
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.llm_utils import LLMConfig, get_http_client

def generate_cluster_name(
    terms: List[str], 
    titles: List[str], 
    language: str = "español"
) -> str:
    """
    Genera un nombre descriptivo para un clúster usando un LLM local o remoto.
    Utiliza las credenciales del archivo .env.
    """
    base_url = LLMConfig.get_auth_url()
    model = LLMConfig.get_model_name()
    
    prompt = f"""Actúa como un experto bibliometrista. Tu tarea es asignar un nombre técnico y descriptivo a un frente de investigación científica.

DATOS DEL CLÚSTER:
- Términos clave (TF-IDF): {', '.join(terms)}
- Títulos representativos:
  {chr(10).join([f"* {t}" for t in titles])}

REGLAS:
1. El nombre debe ser en {language}.
2. Máximo 10 palabras.
3. Debe sonar académico y preciso.
4. No uses frases como "El frente se llama..." o "Basado en...". Solo entrega el nombre.

NOMBRE DEL FRENTE:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Eres un asistente experto en análisis de frentes de investigación científica."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 50
    }

    try:
        # Usamos httpx con verify=False para saltar validación SSL en servidores internos
        with get_http_client(async_mode=False) as client:
            response = client.post(
                f"{base_url}chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                name = result['choices'][0]['message']['content'].strip()
                # Limpiar comillas si el LLM las pone
                name = name.replace('"', '').replace("'", "")
                return name
            else:
                print(f"      [LLM] Error de API: {response.status_code} - {response.text}")
                return "Cluster sin nombre (Error API)"
            
    except Exception as e:
        print(f"      [LLM] Error de conexión: {e}. URL: {base_url}")
        return "Cluster sin nombre (Conexión fallida)"
