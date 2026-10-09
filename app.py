"""Mechanical Failure Detective — local Streamlit + Kev-4B via native llama.cpp System One.

Start: streamlit run app.py
The app never fabricates AI responses: disconnected mode shows only the rule-based baseline.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from time import perf_counter

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import (
    FAULT_LABELS, SCENARIOS, SEVERITY_LABELS, MachineState,
    evaluate_rules, fingerprint, generate_history, state_payload, sweep_states,
)
from model_client import DecisionModelError, create_payload, probe_server, run_inference

st.set_page_config(
    page_title="Mechanical Failure Detective · TXI Lab",
    page_icon="⚙️", layout="wide", initial_sidebar_state="expanded",
)

COLORS = {
    "bearing_wear": "#FC9B6C",
    "lubrication": "#F6C56B",
    "misalignment": "#A9A1FF",
    "imbalance": "#61B6FF",
    "normal": "#4CD6B3",
    "uncertain": "#C1C9D2",
}

st.markdown("""
<style>
    .block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1550px;}
    [data-testid="stMetric"] {background:#122431;border:1px solid #263F4D;border-radius:12px;padding:16px;}
    [data-testid="stMetricLabel"] {color:#B0C4D1;}
    .hero {padding:21px 25px;border-radius:17px;border:1px solid #294752;
      background:linear-gradient(110deg,#102B35,#10202E 63%,#1C2933);margin-bottom:15px;}
    .eyebrow {font-size:0.73rem;letter-spacing:0.16rem;color:#59D5BF;font-weight:700;}
    .hero h1 {margin:4px 0 4px;font-size:2.1rem;font-weight:750;}
    .subtext {color:#A7BCCB;line-height:1.4;font-size:0.96rem;}
    .tag {font-size:0.71rem;letter-spacing:.07rem;padding:4px 9px;border:1px solid #458071;
       border-radius:99px;color:#90E8D6;display:inline-block;margin-top:11px;}
    .sectionhead {font-size:1rem;font-weight:700;color:#D9E8F2;margin:16px 0 8px;}
    .fineprint {font-size:0.78rem;color:#849AAA;}
    div[data-testid="stSidebar"] {border-right:1px solid #274151;}
</style>
""", unsafe_allow_html=True)


def _fill_controls(scenario_key: str) -> None:
    p = SCENARIOS[scenario_key].state
    st.session_state.update({
        "temp": int(p.temperature_c or 0), "vib": float(p.vibration_mm_s or 0),
        "rpm": int(p.rpm), "load": int(p.load_pct),
        "axial": float(p.axial_index), "one_x": float(p.one_x_index),
        "two_x": float(p.two_x_index), "hf": float(p.hf_index),
        "note": p.observation, "maint": p.maintenance, "trend": p.trend,
        "temp_missing": False, "vib_missing": False,
        "_revealed": False,
    })


def current_state() -> MachineState:
    k = st.session_state
    return MachineState(
        None if k["temp_missing"] else float(k["temp"]),
        None if k["vib_missing"] else float(k["vib"]),
        int(k["rpm"]), int(k["load"]), float(k["axial"]),
        float(k["one_x"]), float(k["two_x"]), float(k["hf"]),
        str(k["note"]), str(k["maint"]), str(k["trend"]),
    )


def plot_trends(state: MachineState, trend_strength: float, seed: int) -> go.Figure:
    hist = generate_history(state, trend_strength, seed)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    xs = [r["minute"] for r in hist]
    fig.add_trace(go.Scatter(
        x=xs, y=[r["temperature_c"] for r in hist], name="Temperature (°C)",
        line={"color": "#F7B878", "width": 3},
        fill="tozeroy", fillcolor="rgba(247,184,120,0.055)", mode="lines",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=xs, y=[r["vibration_mm_s"] for r in hist], name="Vibration (mm/s RMS)",
        line={"color": "#55CCBD", "width": 3}, mode="lines",
    ), secondary_y=True)
    fig.update_layout(
        template="plotly_dark", height=295, margin=dict(t=20,b=45,l=5,r=5),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=1.17,x=0),
        hovermode="x unified", font={"color": "#CBDCE5"},
    )
    fig.update_xaxes(title_text="Minutes before current observation", showgrid=True, gridcolor="#223A48")
    fig.update_yaxes(title_text="°C", secondary_y=False, showgrid=True, gridcolor="#223A48", rangemode="tozero")
    fig.update_yaxes(title_text="mm/s", secondary_y=True, showgrid=False, rangemode="tozero")
    return fig


def bar_chart(scores: dict[str, float], *, is_probability: bool, height: int = 290) -> go.Figure:
    order = sorted(scores, key=scores.get, reverse=True)
    values = [scores[k] * (100 if is_probability else 1) for k in order]
    labels = [FAULT_LABELS[k] for k in order]
    colors = [COLORS[k] for k in order]
    fig = go.Figure(go.Bar(
        y=labels[::-1], x=values[::-1], orientation="h", marker_color=colors[::-1],
        text=[f"{v:.1f}{'%' if is_probability else ' pts'}" for v in values[::-1]],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(t=6,l=10,b=30,r=62),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 107 if is_probability else max(6,max(values)*1.2)],
                   title="Probability (%)" if is_probability else "Heuristic points (not probabilities)",
                   gridcolor="#263845"),
        yaxis=dict(tickfont={"size": 11}),
        font={"color": "#D6E5ED", "size": 11}, showlegend=False,
    )
    return fig


def spectrum_fig(state: MachineState) -> go.Figure:
    labels = ["1× shaft", "2× shaft", "Axial", "Bearing HF"]
    vals = [state.one_x_index, state.two_x_index, state.axial_index, state.hf_index]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=["#5CA8ED", "#A79BF6", "#69CFBD", "#FCAC75"]))
    fig.update_layout(
        template="plotly_dark", height=230, margin=dict(t=5,l=0,b=26,r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,10.7], title="Relative index (0–10)",gridcolor="#263845"),
        font={"color": "#C8D6DF"}, showlegend=False,
    )
    return fig


def _record_result(scenario_key: str, state: MachineState, decision, baseline: dict, elapsed: float) -> None:
    if "runs" not in st.session_state:
        st.session_state["runs"] = []
    st.session_state["runs"].append({
        "time": datetime.now().isoformat(timespec="seconds"), "scenario": scenario_key,
        "simulated_truth": SCENARIOS[scenario_key].truth,
        "model_choice": decision.choice, "model_confidence": decision.confidence,
        "model_selected_probability": decision.probabilities.get(decision.choice),
        "model_severity": decision.severity_score,
        "model_inspection_probability": decision.inspect_probability,
        "baseline_choice": baseline["choice"], "model_correct": decision.choice == SCENARIOS[scenario_key].truth,
        "rule_correct": baseline["choice"] == SCENARIOS[scenario_key].truth,
        "elapsed_s": round(elapsed, 3),
        "temperature_c": state.temperature_c, "vibration_mm_s": state.vibration_mm_s,
        "measurement_flags": len(baseline["flags"]),
    })


st.markdown("""
<div class="hero">
  <div class="eyebrow">TXI ENGINEERING LAB / LOCAL DECISION INTELLIGENCE</div>
  <h1>⚙️ Mechanical Failure Detective</h1>
  <div class="subtext">Inject mechanical faults. Compare deterministic diagnostics with Kev-4B's native
  System One judgments. Challenge the model with incomplete and contradictory evidence.</div>
  <span class="tag">SYNTHETIC TELEMETRY · NO REAL EQUIPMENT CONTROL</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔧 Test bench")
    scenario_key = st.selectbox(
        "Inject a scenario", list(SCENARIOS),
        format_func=lambda key: SCENARIOS[key].title, key="scenario_selector",
    )
    if st.session_state.get("_loaded_key") != scenario_key:
        _fill_controls(scenario_key)
        st.session_state["_loaded_key"] = scenario_key

    st.caption(SCENARIOS[scenario_key].subtitle)
    st.markdown("**Telemetry controls**")
    st.slider("Bearing temperature · °C", 20, 170, key="temp")
    st.slider("Vibration RMS · mm/s", 0.0, 16.0, step=0.1, key="vib")
    st.slider("Shaft speed · RPM", 200, 4000, step=25, key="rpm")
    st.slider("Load · %", 0, 100, key="load")
    with st.expander("Advanced sensor features", expanded=False):
        st.slider("1× shaft-order index", 0.0, 10.0, step=0.1, key="one_x")
        st.slider("2× shaft-order index", 0.0, 10.0, step=0.1, key="two_x")
        st.slider("Axial vibration index", 0.0, 10.0, step=0.1, key="axial")
        st.slider("Bearing high-frequency index", 0.0, 10.0, step=0.1, key="hf")
    with st.expander("Operator report / missing data", expanded=False):
        st.text_area("Operator observations", height=95, key="note")
        st.text_input("Maintenance history", key="maint")
        st.text_input("Recent trend", key="trend")
        st.checkbox("Temperature sensor missing", key="temp_missing")
        st.checkbox("Vibration sensor missing", key="vib_missing")

    st.divider()
    st.markdown("### 🧠 Local inference")
    url = st.text_input("llama.cpp server", value="http://127.0.0.1:8080")
    model_id = st.text_input("Model alias (optional)", placeholder="Leave blank for single-model server")
    timeout = st.slider("Inference timeout · seconds", 10, 120, 60, 5)
    if st.button("Check server", use_container_width=True):
        try:
            st.session_state["server_health"] = probe_server(url)
        except DecisionModelError as exc:
            st.session_state["server_health"] = {"healthy": False, "status": str(exc), "models": []}
    health = st.session_state.get("server_health")
    if health:
        if health["healthy"]:
            st.success("llama.cpp responding")
            for m in health["models"]:
                st.caption(f"{m['id']} · {'decision model' if m['decision_capable'] else 'capability not confirmed'}")
        else:
            st.warning("llama.cpp not reachable")
            st.caption(health["status"][:160])
    else:
        st.caption("Not checked. The deterministic baseline works offline.")

state = current_state()
baseline = evaluate_rules(state)
scenario = SCENARIOS[scenario_key]
sig = fingerprint(state)

m1, m2, m3, m4 = st.columns(4)
m1.metric("BEARING TEMPERATURE", "Missing" if state.temperature_c is None else f"{state.temperature_c:.0f} °C")
m2.metric("OVERALL VIBRATION", "Missing" if state.vibration_mm_s is None else f"{state.vibration_mm_s:.1f} mm/s")
m3.metric("SHAFT SPEED", f"{state.rpm:,} RPM")
m4.metric("EQUIPMENT LOAD", f"{state.load_pct}%")

if baseline["flags"]:
    st.warning("**Input validation:** " + " | ".join(baseline["flags"]) + " These are demo sanity checks, not certified machine protection limits.")

tab_diag, tab_sweep, tab_log, tab_notes = st.tabs([
    "📡 Diagnostic console", "🧪 Sensitivity lab", "📋 Experiment log", "📖 Engineering notes",
])

with tab_diag:
    left, right = st.columns([1.18, 1], gap="large")
    with left:
        st.markdown('<div class="sectionhead">Sensor history · last 120 minutes</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_trends(state, scenario.trend_strength, seed=17), use_container_width=True)
        st.markdown('<div class="sectionhead">Extracted vibration signatures</div>', unsafe_allow_html=True)
        st.plotly_chart(spectrum_fig(state), use_container_width=True)
        st.caption("Trend values are synthetically generated and reproducible. The 0–10 indices are NOT measured spectra.")
        with st.expander("Exact System One request payload"):
            st.json(create_payload(state, model_id))

    with right:
        st.markdown('<div class="sectionhead">Decision engine</div>', unsafe_allow_html=True)
        if st.button("▶ Analyze with local Kev-4B", type="primary", use_container_width=True):
            with st.spinner("Requesting real native System One inference from llama.cpp…"):
                try:
                    start = perf_counter()
                    result = run_inference(state, url, model_id, timeout)
                    elapsed = perf_counter() - start
                    st.session_state["last_decision"] = result
                    st.session_state["last_signature"] = sig
                    st.session_state["last_server_config"] = (url, model_id)
                    st.session_state["last_runtime"] = elapsed
                    _record_result(scenario_key, state, result, baseline, elapsed)
                    st.session_state.pop("inference_error", None)
                except DecisionModelError as exc:
                    st.session_state["inference_error"] = str(exc)
                    st.session_state.pop("last_decision", None)
        if st.session_state.get("inference_error"):
            st.error(st.session_state["inference_error"])
        result = st.session_state.get("last_decision")
        fresh = (result is not None and st.session_state.get("last_signature") == sig
                 and st.session_state.get("last_server_config") == (url, model_id))
        if result is not None and not fresh:
            st.info("Telemetry or notes changed. The previous model output is stale; run analysis again.")
        if fresh:
            st.success(f"LIVE MODEL RESULT  ·  {result.model}  ·  {st.session_state.get('last_runtime', 0):.2f} s")
            st.markdown(f"**Top hypothesis:** {FAULT_LABELS[result.choice]}")
            st.plotly_chart(bar_chart(result.probabilities, is_probability=True), use_container_width=True)
            aa, bb, cc = st.columns(3)
            aa.metric("Decision confidence", "—" if result.confidence is None else f"{result.confidence:.0%}")
            bb.metric("Severity · 0–4", "—" if result.severity_score is None else f"{result.severity_score:.2f}")
            cc.metric("Inspect · P(yes)", "—" if result.inspect_probability is None else f"{result.inspect_probability:.0%}")
            st.caption(f"System One probability distribution · {result.input_tokens or '—'} input tokens. These values are not calibrated diagnostic risks.")
            if result.severity_probabilities:
                with st.expander("Severity-level distribution"):
                    st.dataframe(pd.DataFrame([
                        {"Level": f"{int(k)} – {SEVERITY_LABELS[int(k)]}", "Probability": v}
                        for k,v in sorted(result.severity_probabilities.items())
                    ]), hide_index=True, use_container_width=True)
        elif not st.session_state.get("inference_error"):
            st.info("No live model decision for these measurements. Click **Analyze** to query your local Kev-4B server.")

        st.markdown('<div class="sectionhead">Interpretable engineering baseline</div>', unsafe_allow_html=True)
        st.caption("Local heuristic (no AI inference). Points are not probabilities.")
        st.markdown(f"**Baseline hypothesis:** {FAULT_LABELS[baseline['choice']]}")
        st.plotly_chart(bar_chart(baseline["points"], is_probability=False, height=255), use_container_width=True)
        with st.expander("Why the baseline voted this way", expanded=False):
            for reason in baseline["reasons"]:
                st.markdown("- " + reason)

    st.divider()
    cr_left, cr_right = st.columns([1.3, 1])
    with cr_left:
        st.markdown("#### 🎯 The engineering challenge")
        st.write("Try changing **temperature without vibration**, then vary the 1× and 2× indices. Which method changes its answer, and why?")
        st.write("Next, inject the *Faulty temperature sensor* case. A robust system should favor instrument validation over invented mechanical certainty.")
    with cr_right:
        if st.button("Reveal simulated ground truth", use_container_width=True):
            st.session_state["_revealed"] = True
        if st.session_state.get("_revealed"):
            truth = FAULT_LABELS[scenario.truth]
            st.info(f"**Injected condition:** {truth}. {scenario.explanation}")
            if fresh:
                st.write("Model:", "✅ Matched injected label" if result.choice == scenario.truth else "❌ Different from injected label")
            st.write("Rule baseline:", "✅ Matched injected label" if baseline["choice"] == scenario.truth else "❌ Different from injected label")

with tab_sweep:
    st.markdown("### Sensitivity experiment")
    st.write("Hold every measurement fixed except one variable. Compare how the **rule baseline** and the **real model distribution** respond. This can reveal overconfidence and unstable decision boundaries.")
    axis = st.radio("Sweep variable", ["temperature_c", "vibration_mm_s"], horizontal=True,
                    format_func=lambda v: "Temperature · °C" if v == "temperature_c" else "Vibration · mm/s")
    lo, hi, step = (40.0, 110.0, 1.0) if axis == "temperature_c" else (0.5, 12.0, 0.1)
    low, high = st.slider("Range", min_value=lo, max_value=hi,
                          value=(55.0, 105.0) if axis == "temperature_c" else (1.0, 11.0),
                          step=step)
    samples = st.slider("Number of steps", 3, 6, 5)
    values = [round(low + (high-low)*i/(samples-1), 2) for i in range(samples)]
    sweep = sweep_states(state, axis, values)
    df_rules = pd.DataFrame([
        {"Value": v, **evaluate_rules(s)["points"]} for v, s in zip(values, sweep)
    ])
    fig = go.Figure()
    for key, label in FAULT_LABELS.items():
        fig.add_trace(go.Scatter(x=df_rules["Value"], y=df_rules[key],
                   mode="lines+markers", name=label, line={"color": COLORS[key]}))
    fig.update_layout(template="plotly_dark", height=330, margin=dict(t=20,l=10,r=0,b=30),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Temperature (°C)" if axis == "temperature_c" else "Vibration (mm/s)",
                      yaxis_title="Rule points (not probabilities)",
                      legend=dict(orientation="h", y=-0.35), font={"color": "#C6D6E0"})
    st.markdown("**Rule baseline response · instant, offline**")
    st.plotly_chart(fig, use_container_width=True)
    if st.button(f"Run {samples} live Kev-4B fault decisions", type="primary"):
        outcomes = []
        progress = st.progress(0, text="Running local inference…")
        try:
            for idx, (v, ss) in enumerate(zip(values, sweep)):
                decision = run_inference(ss, url, model_id, timeout, only_fault=True)
                outcomes.append({"Value": v, **decision.probabilities})
                progress.progress((idx+1)/samples, text=f"Completed {idx+1}/{samples} decisions")
            st.session_state["sweep"] = outcomes
            st.session_state["sweep_sig"] = (sig, axis, low, high, samples, url, model_id)
            st.session_state.pop("sweep_error", None)
        except DecisionModelError as exc:
            st.session_state["sweep_error"] = str(exc)
            st.session_state.pop("sweep", None)
        finally:
            progress.empty()
    if st.session_state.get("sweep_error"):
        st.error(st.session_state["sweep_error"])
    if "sweep" in st.session_state:
        if st.session_state.get("sweep_sig") != (sig, axis, low, high, samples, url, model_id):
            st.info("The previous sensitivity experiment is stale. Run the sweep again.")
        else:
            data = pd.DataFrame(st.session_state["sweep"])
            fig2 = go.Figure()
            for key, label in FAULT_LABELS.items():
                fig2.add_trace(go.Scatter(x=data["Value"], y=data[key]*100, name=label,
                                          mode="lines+markers", line={"color": COLORS[key]}))
            fig2.update_layout(template="plotly_dark", height=360,
                margin=dict(t=20,l=10,r=0,b=30), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="Model probability (%)", range=[0,100]),
                xaxis_title="Temperature (°C)" if axis == "temperature_c" else "Vibration (mm/s)",
                legend=dict(orientation="h", y=-0.35),font={"color": "#C6D6E0"})
            st.markdown("**Kev-4B response · actual inference**")
            st.plotly_chart(fig2, use_container_width=True)
            st.download_button("Download sweep data (CSV)", data=data.to_csv(index=False),
                               file_name="systemone_sensitivity.csv", mime="text/csv")

with tab_log:
    st.markdown("### Experiment log")
    runs = st.session_state.get("runs", [])
    if runs:
        df = pd.DataFrame(runs)
        a,b,c = st.columns(3)
        a.metric("Completed model decisions", len(df))
        b.metric("Match to injected label", f"{df['model_correct'].mean():.0%}")
        c.metric("Baseline label matches", f"{df['rule_correct'].mean():.0%}")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Download experiment log (CSV)", df.to_csv(index=False),
                           file_name="mechanical_failure_experiments.csv", mime="text/csv")
        if st.button("Clear log"):
            st.session_state["runs"] = []
            st.rerun()
    else:
        st.info("Run a few scenarios with the model to collect an in-session comparison log.")
    st.caption("These scores compare against synthetic labels only. They are not estimates of real diagnostic accuracy.")

with tab_notes:
    st.markdown("### What the demo teaches")
    st.markdown("""
1. **Fault classification (choice):** pick one hypothesis from six explicit categories.
2. **Condition severity (score):** estimate a continuous ordinal rating from 0 to 4.
3. **Inspection question (noul):** assign a probability to a yes/no decision.
4. **Engineering baseline:** transparent heuristics serve as a comparator, not ground truth.
5. **Data quality:** missing or implausible measurements should prompt verification, not automatic action.
6. **Sensitivity:** learned decisions can change sharply as measurements cross thresholds.

**Limitations:** all telemetry, spectral indices, trend histories, and injected faults are synthetic. Model confidence is not a calibrated probability of a real mechanical failure. The rules and their thresholds are illustrative, not standards or approved maintenance criteria. Human technicians and validated safety processes must govern inspection and equipment operation.

**Privacy:** the dashboard only permits a loopback HTTP inference server. No cloud model service is called by the Python code.
""")
    st.markdown("**Suggested 30-minute workshop:** 5 minutes on sensor physics; 10 minutes testing presets; 10 minutes crafting a misleading scenario; 5 minutes on how to evaluate failures.")
    with st.expander("Current machine state as JSON"):
        st.code(json.dumps(state_payload(state), indent=2), language="json")
    st.caption("System One interface: https://github.com/ggml-org/llama.cpp/tree/master/tools/server")
