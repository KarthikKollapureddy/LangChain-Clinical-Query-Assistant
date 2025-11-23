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
        """
        Call WAIP completion skill and return a plain text response.
        """
        url = f"{self.base_url}/v1.1/skills/completion/query"
        payload = {
            "messages": [{"content": prompt, "role": "user"}],
            "skill_parameters": {"model_name": model_name, "max_output_tokens": max_output_tokens},
            "stream_response": False,
        }
        resp = requests.post(url, json=payload, headers=self.headers, verify=False)
        if resp.status_code == 422:
            # try an alternate minimal payload seen to work in some WAIP deployments
            alt_payload = {"input": prompt, "model": model_name}
            resp = requests.post(url, json=alt_payload, headers=self.headers, verify=False)

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            # include response body to make debugging WAIP 422 responses easier
            body = None
            try:
                body = resp.text
            except Exception:
                body = '<no response body>'
            raise RuntimeError(f"WAIP API error {resp.status_code}: {body}")

        # parse common JSON shapes and return a single text string
        try:
            j = resp.json()
        except ValueError:
            return resp.text

        # common WAIP-like response: {"data": {"content": "..."}}
        if isinstance(j, dict):
            data = j.get("data")
            if isinstance(data, dict) and "content" in data:
                return data.get("content")
            # sometimes the top-level has 'content'
            if "content" in j:
                return j.get("content")

        # fallback to text representation
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
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            # bubble up so caller can decide fallback; include response text for debugging
            raise

        # return raw json when available — caller decides how to interpret
        try:
            return resp.json()
        except ValueError:
            return resp.text

__all__ = ["WAIPClient"]
