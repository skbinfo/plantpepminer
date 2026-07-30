import json
import re
from pathlib import Path
from typing import Any
from urllib import error, request

def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")
    return parsed

def validate_verification_response(data: dict[str, Any]) -> dict[str, Any]:
    if "valid_sequence" not in data or not isinstance(data["valid_sequence"], bool):
        raise ValueError("Response missing boolean 'valid_sequence'")
    
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise ValueError("Confidence must be a numeric value")
    
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")
        
    support_type = str(data.get("support_type", ""))
    if support_type not in ("strong", "moderate", "weak", "reject"):
        raise ValueError("support_type must be strong, moderate, weak, or reject")
    
    linked_techniques = data.get("linked_techniques", [])
    if not isinstance(linked_techniques, list):
        raise ValueError("linked_techniques must be a list of technique names")
    
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
        
    return {
        "valid_sequence": data["valid_sequence"],
        "confidence": confidence,
        "support_type": support_type,
        "linked_techniques": [str(t) for t in linked_techniques],
        "reason": str(reason),
        "species": str(data.get("species", "")) if data.get("species") else None,
        "is_synthesized": bool(data.get("is_synthesized", False)),
        "synthesized_line": str(data.get("synthesized_line", "")) if data.get("synthesized_line") else None
    }

class LocalLLMClient:
    def __init__(self, model: str, endpoint: str, timeout: int) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 512, "num_thread": 6},
        }).encode("utf-8")
        
        req = request.Request(
            f"{self.endpoint}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return str(res_data.get("response", ""))
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama endpoint {self.endpoint}/api/generate: {exc}") from exc

def pull_model(client: LocalLLMClient) -> None:
    body = json.dumps({"name": client.model}).encode("utf-8")
    req = request.Request(
        f"{client.endpoint}/api/pull",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    print(f"Ensuring model {client.model} is pulled (this may take a while if not cached)...")
    try:
        with request.urlopen(req, timeout=600) as response:
            pass
    except error.URLError as exc:
        print(f"Warning: Could not auto-pull model {client.model}. Error: {exc}")


def validate_technique_verification_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validates LLM JSON response for technique verification."""
    is_confirmed = data.get("technique_confirmed")
    if not isinstance(is_confirmed, bool):
        raise ValueError("Response missing boolean 'technique_confirmed'")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise ValueError("Confidence must be a numeric value")

    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    evidence_line = data.get("evidence_line", "")
    if not isinstance(evidence_line, str):
        raise ValueError("evidence_line must be a string")

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    return {
        "technique_confirmed": is_confirmed,
        "confidence": confidence,
        "evidence_line": evidence_line,
        "reason": reason,
    }


def validate_species_verification_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validates LLM JSON response for species verification."""
    is_confirmed = data.get("species_confirmed")
    if not isinstance(is_confirmed, bool):
        raise ValueError("Response missing boolean 'species_confirmed'")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise ValueError("Confidence must be a numeric value")

    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    verified_species = data.get("verified_species")
    if verified_species is not None and not isinstance(verified_species, str):
        raise ValueError("verified_species must be a string or null")

    evidence_line = data.get("evidence_line", "")
    if not isinstance(evidence_line, str):
        raise ValueError("evidence_line must be a string")

    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    return {
        "species_confirmed": is_confirmed,
        "confidence": confidence,
        "verified_species": verified_species,
        "evidence_line": evidence_line,
        "reason": reason,
    }
