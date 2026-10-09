"""Deterministic educational bearing fault simulator and heuristic baseline.

These signals and labels are SYNTHETIC, not validated predictive maintenance data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import random
from typing import Optional


FAULT_LABELS = {
    "bearing_wear": "Bearing wear",
    "lubrication": "Lubrication problem",
    "misalignment": "Shaft misalignment",
    "imbalance": "Rotor imbalance",
    "normal": "Normal operation",
    "uncertain": "Insufficient / conflicting evidence",
}

SEVERITY_LABELS = [
    "Normal / monitor",
    "Minor deviation",
    "Plan inspection",
    "Prompt inspection",
    "Potentially critical – follow procedures",
]


@dataclass(frozen=True)
class MachineState:
    temperature_c: Optional[float]
    vibration_mm_s: Optional[float]
    rpm: int
    load_pct: int
    axial_index: float
    one_x_index: float
    two_x_index: float
    hf_index: float
    observation: str
    maintenance: str
    trend: str


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    subtitle: str
    truth: str
    explanation: str
    state: MachineState
    trend_strength: float


SCENARIOS = {
    s.key: s
    for s in [
        Scenario(
            "healthy", "Normal operation", "A healthy reference bearing", "normal",
            "Stable conditions and low diagnostic vibration indices.",
            MachineState(49, 1.2, 1750, 58, 0.6, 0.8, 0.4, 0.5,
                         "Normal motor hum", "Routine lubrication 2 weeks ago", "Stable over 30 days"),
            0.0,
        ),
        Scenario(
            "bearing", "Bearing race deterioration", "Clicking + high-frequency vibration", "bearing_wear",
            "Localized bearing defects can produce periodic impacts and strong high-frequency vibration.",
            MachineState(84, 7.8, 1750, 68, 1.7, 2.4, 1.5, 8.2,
                         "Periodic metallic clicking; envelope spectrum shows defect harmonics",
                         "Lubrication completed 10 days ago", "Impact vibration rising over 72 hours"),
            1.0,
        ),
        Scenario(
            "lubrication", "Lubrication starvation", "Rising heat + squeal", "lubrication",
            "Poor lubrication often produces friction, increasing heat and broadband vibration.",
            MachineState(94, 5.2, 1750, 66, 1.3, 1.7, 0.8, 4.9,
                         "High-pitched squealing under load",
                         "Lubrication overdue by 90 days", "Steadily rising temperature over 4 days"),
            1.0,
        ),
        Scenario(
            "misalignment", "Coupling misalignment", "Axial and 2× shaft vibration", "misalignment",
            "Misalignment can elevate axial motion and harmonics of shaft frequency, particularly after reassembly.",
            MachineState(71, 7.0, 1750, 70, 7.2, 3.5, 7.0, 1.4,
                         "Low-frequency rumble; unusually strong 2× shaft frequency",
                         "Coupling replaced yesterday", "Vibration rose immediately after maintenance"),
            0.5,
        ),
        Scenario(
            "imbalance", "Rotor imbalance", "Strong 1× radial vibration", "imbalance",
            "Rotor imbalance commonly causes strong vibration synchronized to shaft rotation.",
            MachineState(60, 7.5, 1750, 70, 1.0, 8.4, 0.9, 0.7,
                         "Steady cyclic hum at shaft rotation frequency",
                         "Rotor cleaned last shift", "Abrupt change after rotor cleaning"),
            0.3,
        ),
        Scenario(
            "load", "Ambiguous high-load heat", "Hot but mechanically quiet", "uncertain",
            "High load can raise temperature without a mechanical defect. Thermal context and extra testing matter.",
            MachineState(89, 2.1, 1760, 98, 0.8, 1.2, 0.6, 0.7,
                         "No unusual sound; load near full capacity",
                         "Normal recent inspection", "Temperature follows production loading"),
            0.6,
        ),
        Scenario(
            "sensor", "Faulty temperature sensor", "Abrupt implausible spike", "uncertain",
            "The extreme temperature conflicts with the other evidence; validate instrumentation first.",
            MachineState(155, 1.0, 1750, 56, 0.5, 0.8, 0.5, 0.5,
                         "No heat felt at housing; no unusual noise",
                         "Temperature probe replaced today", "Temperature jumped 90°C in one sample"),
            0.0,
        ),
    ]
}


def state_payload(state: MachineState) -> dict:
    """The actual information passed to the decision model."""
    d = asdict(state)
    return {
        "equipment": "Synthetic 1750-RPM industrial motor / rolling-element bearing",
        "data_source": "Educational simulation, not physical sensors",
        "sensors": {
            "bearing_temperature_c": d["temperature_c"],
            "overall_vibration_rms_mm_s": d["vibration_mm_s"],
            "shaft_speed_rpm": d["rpm"],
            "motor_load_percent": d["load_pct"],
        },
        "derived_vibration_indices_0_to_10": {
            "axial_vibration": d["axial_index"],
            "one_times_shaft_frequency": d["one_x_index"],
            "two_times_shaft_frequency": d["two_x_index"],
            "high_frequency_bearing_impacts": d["hf_index"],
        },
        "operator_observation": d["observation"],
        "maintenance_context": d["maintenance"],
        "recent_trend": d["trend"],
        "measurement_note": "Indices are fictional 0-10 diagnostic features, NOT physical engineering units.",
    }


def fingerprint(state: MachineState) -> str:
    raw = json.dumps(state_payload(state), sort_keys=True, ensure_ascii=True)
    return sha256(raw.encode("utf-8")).hexdigest()


def validate(state: MachineState) -> list[str]:
    """Educational sanity checks; thresholds are illustrative, NOT safety limits."""
    flags = []
    if state.temperature_c is None:
        flags.append("Temperature sensor missing: thermal inference is incomplete.")
    elif state.temperature_c < -20 or state.temperature_c > 130:
        flags.append("Temperature outside this demo's plausible range: verify the probe.")
    if state.vibration_mm_s is None:
        flags.append("Overall vibration missing: classification evidence is incomplete.")
    elif state.vibration_mm_s < 0 or state.vibration_mm_s > 30:
        flags.append("Overall vibration outside this demo's expected range: verify the reading.")
    if state.rpm < 200 or state.rpm > 4000:
        flags.append("Speed outside the demo motor's usual operating envelope.")
    if state.temperature_c is not None and state.vibration_mm_s is not None:
        if state.temperature_c > 130 and state.vibration_mm_s < 2.5:
            flags.append("Hot reading conflicts with otherwise quiet vibration: cross-check measurements.")
    return flags


def _above(value: Optional[float], threshold: float) -> float:
    return float(value is not None and value > threshold)


def evaluate_rules(state: MachineState) -> dict:
    """Intentionally simple interpretable baseline. Points are NOT probabilities."""
    scores = {name: 0.2 for name in FAULT_LABELS}
    reasons: list[str] = []
    flags = validate(state)
    obs = (state.observation + " " + state.maintenance).lower()
    temp = state.temperature_c
    vib = state.vibration_mm_s

    if temp is not None and temp > 77:
        scores["lubrication"] += 1.0
        scores["bearing_wear"] += 0.7
        reasons.append("Elevated temperature is compatible with friction-related faults.")
    if _above(vib, 4):
        scores["bearing_wear"] += 0.5
        scores["misalignment"] += 0.6
        scores["imbalance"] += 0.6
        reasons.append("Overall vibration is elevated but does not identify a unique root cause.")
    if state.hf_index > 4:
        scores["bearing_wear"] += 2.3
        scores["lubrication"] += 0.6
        reasons.append("High-frequency impact index supports a bearing-related problem.")
    if state.axial_index > 4 and state.two_x_index > 3.5:
        scores["misalignment"] += 3.2
        reasons.append("High axial and 2× vibration indices support misalignment.")
    if state.one_x_index > 4.5 and state.two_x_index < 2:
        scores["imbalance"] += 3.4
        reasons.append("Dominant 1× vibration with weak 2× supports imbalance.")
    if any(x in obs for x in ("click", "defect harmonic", "metallic")):
        scores["bearing_wear"] += 2.2
        reasons.append("Periodic clicking or defect harmonics suggest rolling-element damage.")
    if any(x in obs for x in ("squeal", "lubrication overdue")):
        scores["lubrication"] += 2.5
        reasons.append("Squeal / overdue lubrication supports lubricant starvation.")
    if any(x in obs for x in ("coupling replaced", "motor replaced", "reassembly")):
        scores["misalignment"] += 1.8
        reasons.append("Recent drivetrain service increases the plausibility of misalignment.")
    if "rotor cleaned" in obs:
        scores["imbalance"] += 1.7
        reasons.append("A change after rotor cleaning can indicate mass imbalance.")
    if vib is not None and vib < 2.5 and state.hf_index < 2 and state.axial_index < 2:
        scores["normal"] += 2.4
        reasons.append("Vibration indices are mechanically quiet.")
        if state.load_pct >= 90 and temp is not None and temp > 80:
            scores["uncertain"] += 2.8
            reasons.append("High heat under high load is ambiguous without a baseline at this duty cycle.")
    if temp is not None and temp < 65 and vib is not None and vib < 2.5:
        scores["normal"] += 1.5
    if flags:
        scores["uncertain"] += 9.0
        reasons.append("Data quality flags take precedence over diagnostic hypotheses.")
    if not reasons:
        scores["uncertain"] += 1.0
        reasons.append("No strong discriminating evidence was found.")
    rank = sorted(scores.items(), key=lambda it: it[1], reverse=True)
    return {"choice": rank[0][0], "points": scores, "reasons": reasons, "flags": flags}


def generate_history(state: MachineState, trend_strength: float, seed: int = 42) -> list[dict]:
    """Synthetic 2-hour trend, ending exactly at the current control values."""
    rng = random.Random(seed)
    points = []
    for n in range(49):
        phase = n / 48.0
        tmin = -120 + n * 2.5
        heat = state.temperature_c if state.temperature_c is not None else None
        vib = state.vibration_mm_s if state.vibration_mm_s is not None else None
        if n == 48:
            tt, vv = heat, vib
        else:
            tt = None if heat is None else max(0.0, heat - 13 * trend_strength * (1 - phase) + rng.gauss(0, 0.85))
            vv = None if vib is None else max(0.0, vib - 2.8 * trend_strength * (1 - phase) + rng.gauss(0, 0.17))
        points.append({"minute": tmin, "temperature_c": tt, "vibration_mm_s": vv})
    return points


def sweep_states(state: MachineState, axis: str, values: list[float]) -> list[MachineState]:
    if axis not in ("temperature_c", "vibration_mm_s"):
        raise ValueError("Unsupported sweep variable")
    return [replace(state, **{axis: value}) for value in values]
