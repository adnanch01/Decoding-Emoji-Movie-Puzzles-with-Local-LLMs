import requests
import time
import json

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def query_ollama(model: str, prompt: str, temperature: float = 0.3, top_p: float = 0.9):
    """
    Sends a chat request to the local Ollama server and returns (full_text_response, latency).
    Handles Ollama's newline-delimited JSON streaming format.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "top_p": top_p}
    }

    t0 = time.time()
    # streaming=False still returns all ndjson lines in one go; we parse manually
    response = requests.post(OLLAMA_API_URL, json=payload)
    latency = time.time() - t0

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    content_parts = []
    for line in response.text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = data.get("message", {}).get("content")
        if msg:
            content_parts.append(msg)

    content = "".join(content_parts).strip()
    return content, latency
