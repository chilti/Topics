import os
import httpx
from dotenv import load_dotenv

# Aseguramos carga de entorno
load_dotenv()

class LLMConfig:
    @staticmethod
    def get_auth_url():
        """Construye la URL con Basic Auth para LM Studio/Remote LLM."""
        user = os.getenv("LLM_USER")
        password = os.getenv("LLM_PASSWORD")
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        
        # Eliminar /v1 si está al final, ya que el cliente de OpenAI o requests lo suele añadir o necesitamos control
        # Sin embargo, el orchestrator de RAGs lo mantiene. Vamos a imitar su comportamiento.
        
        if not base_url.endswith("/"):
            base_url += "/"
            
        if user and password:
            if "://" in base_url:
                proto, rest = base_url.split("://", 1)
                return f"{proto}://{user}:{password}@{rest}"
            else:
                return f"http://{user}:{password}@{base_url}"
        return base_url

    @staticmethod
    def get_model_name(default="openai/gpt-oss-20b"):
        return os.getenv("LLM_MODEL", default)

    @staticmethod
    def get_api_key():
        # LM Studio no suele requerir API Key real, pero a veces se usa como placeholder
        return os.getenv("LLM_API_KEY", "lm-studio")

def get_http_client(async_mode=False, timeout=120):
    """Retorna un cliente httpx configurado para saltar validación SSL."""
    if async_mode:
        return httpx.AsyncClient(verify=False, timeout=timeout)
    return httpx.Client(verify=False, timeout=timeout)
