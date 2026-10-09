"""Command-line real API smoke test — runs without Streamlit."""
from __future__ import annotations
import argparse
from core import FAULT_LABELS, SCENARIOS
from model_client import DecisionModelError, probe_server, run_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Test local Kev-4B System One inference")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="", help="Optional model alias for llama.cpp router mode")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="bearing")
    args = parser.parse_args()
    try:
        status = probe_server(args.url)
        print("Server health:", status["status"])
        if not status["healthy"]:
            print("Inference is unavailable. Start llama.cpp with Kev-4B first.")
            return 1
        decision = run_inference(SCENARIOS[args.scenario].state, args.url, args.model, 90)
        print(f"Scenario: {SCENARIOS[args.scenario].title}")
        print(f"Model: {decision.model}")
        print(f"Selected fault: {FAULT_LABELS[decision.choice]}")
        print("Choice probabilities:")
        for label, prob in sorted(decision.probabilities.items(), key=lambda p: -p[1]):
            print(f"  {FAULT_LABELS[label]:36} {prob:6.1%}")
        print("Severity score (0–4):", decision.severity_score)
        print("Inspection probability:", decision.inspect_probability)
        return 0
    except DecisionModelError as exc:
        print("System One error:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
