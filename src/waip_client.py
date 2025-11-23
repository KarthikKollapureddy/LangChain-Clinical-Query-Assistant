import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

class WAIPClient:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("WAIP_API_KEY")
        self.base_url = base_url or os.getenv("WAIP_API_ENDPOINT", "https://api.waip.wiprocms.com")
        if not self.api_key:
            raise RuntimeError("WAIP_API_KEY not set")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def chat_completion(self, prompt: str, model_name: str = "gpt-4o", max_output_tokens: int = 512):
        url = f"{self.base_url}/v1.1/skills/completion/query"
        payload = {
            "messages": [{"content": prompt, "role": "user"}],
            "skill_parameters": {"model_name": model_name, "max_output_tokens": max_output_tokens},
            "stream_response": False,
        }
        resp = requests.post(url, json=payload, headers=self.headers, verify=False)
        resp.raise_for_status()
        return resp.text

    def embeddings(self, texts, model_name: str = "text-embedding-3-large"):
        # WAIP may or may not support embeddings directly; use completion query with emb_type flag if needed
        url = f"{self.base_url}/v1.1/skills/completion/query"
        payload = {
            "messages": [{"content": t, "role": "user"} for t in texts],
            "skill_parameters": {"model_name": model_name, "emb_type": "openai"},
            "stream_response": False,
        }
        resp = requests.post(url, json=payload, headers=self.headers, verify=False)
        resp.raise_for_status()
        # The response format may vary — return raw text for now
        return resp.text

__all__ = ["WAIPClient"]
