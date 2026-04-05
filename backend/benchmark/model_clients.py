import os
import time
import random
import requests
from typing import Optional

# Activer le mode mock (pas d'appels réels) en définissant MOCK_MODE=true dans .env
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

#HF_API_URL = "https://router.huggingface.co/hf-inference/"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _mock_response(question: str, model_id: str) -> dict:
    """Génère une réponse simulée pour les tests."""
    latency = random.uniform(0.5, 2.0)
    mock_answer = f"Réponse simulée du modèle {model_id} : {question[:100]}…"
    input_tokens = _estimate_tokens(question)
    output_tokens = _estimate_tokens(mock_answer)
    return {
        "text": mock_answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency,
        "error": None,
    }


# Pour Qwen (via Hugging Face Chat API)
def call_huggingface_chat(
    model_id: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> dict:
    if MOCK_MODE:
        return _mock_response(question, model_id)

    if not HF_API_KEY:
        return error_response(0, "HUGGINGFACE_API_KEY manquante")

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.perf_counter() - t0

        if response.status_code != 200:
            return error_response(latency, f"HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        generated = data["choices"][0]["message"]["content"]
        input_tokens = data.get("usage", {}).get("prompt_tokens", _estimate_tokens(question))
        output_tokens = data.get("usage", {}).get("completion_tokens", _estimate_tokens(generated))

        return {
            "text": generated.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
            "error": None,
        }
    except Exception as e:
        return error_response(time.perf_counter() - t0, str(e))


# Pour Mistral API (officielle)
def call_mistral_api(
    model_id: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> dict:
    if MOCK_MODE:
        return _mock_response(question, model_id)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return error_response(0, "MISTRAL_API_KEY manquante")

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.perf_counter() - t0
        if response.status_code != 200:
            return error_response(latency, f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        generated = data["choices"][0]["message"]["content"]
        input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {
            "text": generated.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
            "error": None,
        }
    except Exception as e:
        return error_response(time.perf_counter() - t0, str(e))


# Pour LLaMA via Groq (exemple)
def call_groq_api(
    model_id: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> dict:
    if MOCK_MODE:
        return _mock_response(question, model_id)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return error_response(0, "GROQ_API_KEY manquante")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.perf_counter() - t0
        if response.status_code != 200:
            return error_response(latency, f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        generated = data["choices"][0]["message"]["content"]
        input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {
            "text": generated.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
            "error": None,
        }
    except Exception as e:
        return error_response(time.perf_counter() - t0, str(e))


def error_response(latency: float, error_msg: str) -> dict:
    return {
        "text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": latency,
        "error": error_msg,
    }

def call_model(provider: str, model_id: str, system_prompt: str,
               question: str, max_tokens: int, temperature: float) -> dict:
    if provider == "huggingface_chat":
        return call_huggingface_chat(model_id, system_prompt, question, max_tokens, temperature)
    elif provider == "mistral_api":
        return call_mistral_api(model_id, system_prompt, question, max_tokens, temperature)
    elif provider == "groq_api":
        return call_groq_api(model_id, system_prompt, question, max_tokens, temperature)
    elif provider == "huggingface":
        # fallback si vous voulez garder l'ancien
        return call_huggingface_old(model_id, system_prompt, question, max_tokens, temperature)
    else:
        return error_response(0, f"Provider inconnu : {provider}")