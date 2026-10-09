"""Small native llama.cpp System One HTTP client. No hosted inference or fallback fake predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
import requests

from core import FAULT_LABELS, SEVERITY_LABELS, MachineState, state_payload


class DecisionModelError(RuntimeError):
    """Model server unavailable or response does not match System One contract."""


@dataclass(frozen=True)
class Decision:
    choice: str
    probabilities: dict[str, float]
    confidence: float | None
    severity_score: float | None
    severity_probabilities: dict[str, float]
    inspect_probability: float | None
    input_tokens: int | None
    model: str


def local_server_url(raw: str) -> str:
    """Restrict the laptop demo to localhost for safety and privacy."""
    url = raw.strip().rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise DecisionModelError("Use a local URL such as http://127.0.0.1:8080 (no remote endpoints).")
    try:
        port = parsed.port
    except ValueError as exc:
        raise DecisionModelError("Invalid port in server URL.") from exc
    if parsed.netloc.endswith(":") or (port is not None and not 1 <= port <= 65535):
        raise DecisionModelError("Invalid port in server URL.")
    return url


def create_questions() -> dict[str, Any]:
    return {
        "fault": {
            "type": "choice",
            "instructions": (
                "Determine which single diagnosis is best supported by the observations of this simulated "
                "rotating machine. Consider duty cycle, sensor validity, maintenance events, 1×/2× vibration "
                "and high-frequency impacts. Prefer 'uncertain' when evidence is missing, implausible or mixed. "
                "The readings are simulated and do not imply a verified failure."
            ),
            "criteria": {
                "bearing_wear": "Rolling-element or race deterioration, impacts, clicking, bearing defect harmonics",
                "lubrication": "Inadequate lubrication, excessive friction, heat, squealing or overdue grease",
                "misalignment": "Shaft or coupling offset; abnormal axial and 2× shaft-order vibration",
                "imbalance": "Uneven rotating mass; dominant 1× shaft-order vibration and low axial motion",
                "normal": "Normal operation for current load, with low mechanical vibration",
                "uncertain": "Insufficient, conflicting, invalid or misleading evidence; verify sensors or inspect first",
            },
        },
        "severity": {
            "type": "score",
            "instructions": (
                "How severe is the apparent equipment condition based only on the reported observations? "
                "Treat unknown and corrupt signals cautiously. This is a demonstration classification, "
                "not a safety or shutdown instruction."
            ),
            "criteria": SEVERITY_LABELS,
        },
        "inspect": {
            "type": "noul",
            "instructions": (
                "Do these observations justify a qualified technician checking the machine or measurement "
                "instrumentation according to local maintenance procedures?"
            ),
        },
    }


def create_payload(state: MachineState, model_id: str = "", only_fault: bool = False) -> dict:
    payload = {"state": state_payload(state), "questions": create_questions()}
    if only_fault:
        payload["questions"] = {"fault": payload["questions"]["fault"]}
    if model_id.strip():
        payload["model"] = model_id.strip()
    return payload


def _numeric(value: Any, field: str, *, allow_missing: bool = False) -> float | None:
    if value is None and allow_missing:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionModelError(f"Invalid {field}: expected a number")
    x = float(value)
    if x != x or x in (float("inf"), -float("inf")):
        raise DecisionModelError(f"Invalid {field}: non-finite number")
    return x


def _probs(value: Any, allowed: set[str]) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise DecisionModelError("Missing probability distribution in decision response")
    result = {}
    for key, raw in value.items():
        if key not in allowed:
            continue
        num = _numeric(raw, f"probability {key}")
        if not 0 <= num <= 1:
            raise DecisionModelError(f"Invalid probability for {key}")
        result[key] = num
    if set(result) != allowed:
        raise DecisionModelError("Incomplete probability distribution in decision response")
    if abs(sum(result.values()) - 1) > 0.02:
        raise DecisionModelError("Probability distribution does not approximately sum to 1")
    return result


def parse_decision(body: dict[str, Any], only_fault: bool = False) -> Decision:
    try:
        answers = body["answers"]
        fault = answers["fault"]
        choice = fault["choice"]
        if choice not in FAULT_LABELS:
            raise DecisionModelError(f"Unrecognized fault label: {choice!r}")
        p = _probs(fault["probabilities"], set(FAULT_LABELS))
        confidence = _numeric(fault.get("confidence"), "confidence", allow_missing=True)
        if confidence is not None and not 0 <= confidence <= 1:
            raise DecisionModelError("Model confidence outside 0–1")
        score, severity_probs, inspect = None, {}, None
        if not only_fault:
            s = answers["severity"]
            score = _numeric(s["score"], "score")
            if not 0 <= score <= len(SEVERITY_LABELS) - 1:
                raise DecisionModelError("Severity score out of range")
            severity_probs = _probs(s["probabilities"], {str(i) for i in range(len(SEVERITY_LABELS))})
            inspect = _numeric(answers["inspect"]["noul"], "noul")
            if not 0 <= inspect <= 1:
                raise DecisionModelError("Inspection probability out of range")
        usage = body.get("usage", {})
        tokens = usage.get("input_tokens")
        return Decision(choice, p, confidence, score, severity_probs, inspect,
                        tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None,
                        str(body.get("model", "local model")))
    except (KeyError, TypeError, AttributeError) as exc:
        raise DecisionModelError(f"Unexpected System One response structure: {exc}") from exc


def run_inference(state: MachineState, base_url: str = "http://127.0.0.1:8080",
                  model_id: str = "", timeout: float = 60, only_fault: bool = False) -> Decision:
    url = local_server_url(base_url) + "/v1/systemone"
    payload = create_payload(state, model_id, only_fault)
    try:
        resp = requests.post(url, json=payload, timeout=(3, timeout))
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise DecisionModelError("Model returned a non-object response")
        return parse_decision(data, only_fault)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            msg = "System One endpoint not found (404). Update llama.cpp to a recent build."
        elif status == 501:
            msg = "This server does not support native decision inference for the loaded model (501)."
        else:
            details = (exc.response.text if exc.response is not None else "")[:450]
            msg = f"HTTP {status} from llama.cpp: {details}"
        raise DecisionModelError(msg) from exc
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise DecisionModelError(f"Cannot reach llama.cpp or request timed out: {exc}") from exc
    except requests.exceptions.JSONDecodeError as exc:
        raise DecisionModelError("Server response is not valid JSON") from exc


def probe_server(base_url: str) -> dict:
    """Returns informative health/model capability; health does not prove /v1/systemone works."""
    base = local_server_url(base_url)
    try:
        health_response = requests.get(base + "/health", timeout=(2, 3))
        health_response.raise_for_status()
        health = health_response.json()
        info = {"healthy": True, "status": health.get("status", "ok"), "models": []}
        try:
            models = requests.get(base + "/v1/models", timeout=(2, 3))
            models.raise_for_status()
            for item in models.json().get("data", []):
                info["models"].append({
                    "id": item.get("id", "?"),
                    "decision_capable": "decisions" in item.get("architecture", {}).get("output_modalities", []),
                })
        except (ValueError, requests.RequestException, AttributeError, TypeError):
            pass
        return info
    except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
        return {"healthy": False, "status": str(exc), "models": []}
