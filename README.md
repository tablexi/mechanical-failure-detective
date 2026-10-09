# Mechanical Failure Detective

**A local-first mechanical-engineering workshop demo using Python, Streamlit, Kev-4B, and llama.cpp's native System One API.**

This hands-on lab simulates an industrial motor's rolling-element bearing. Junior engineers can modify operating data, inject synthetic mechanical faults, and compare two very different ways to reason about a machine:

- **Native System One decisions:** the **real** Kev-4B model loaded in llama.cpp returns `choice`, `score`, and `noul` results from `/v1/systemone`.
- **Interpretable engineering baseline:** basic Python point-based rules are always available, including when the model server is offline. Baseline points are **not** probabilities.

There are **no fake model results**. The dashboard only presents model probabilities after a successful local HTTP request.

## What's included

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit control-room-style dashboard |
| `core.py` | 7 synthetic scenarios, sensor history, input quality checks, heuristic baseline |
| `model_client.py` | Validated llama.cpp System One API client |
| `check_server.py` | CLI live-inference smoke test without Streamlit |
| `.streamlit/config.toml` | Dark theme |
| `requirements.txt` | Streamlit, Plotly, pandas and requests |
| `tests/` | Unit tests with mocked HTTP; no model required |

### Scenarios

1. Normal operation
2. Bearing race deterioration (HF impacts, clicking)
3. Lubrication starvation (heat, squeal)
4. Coupling misalignment (axial + 2× vibration)
5. Rotor imbalance (strong 1× vibration)
6. Ambiguous high-load heat (hot, low vibration)
7. Faulty temperature probe (physically contradictory measurements)

You can change temperature, vibration RMS, load, RPM, derived indices, notes, maintenance history, and missing-sensor flags independently. Plotted time histories are **synthetic** and deterministic for a given state.

## Quick start

**Requirements:** Python 3.11+; an up-to-date llama.cpp build with native `/v1/systemone`; and a Kev-4B GGUF model. After model/requirements downloads, everything runs locally. Use two terminals.

### 1. Start llama.cpp with Kev-4B

If Kev-4B is already on your computer:

```bash
llama-server -m /absolute/path/to/your/Kev-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080
```

Or use llama.cpp's managed model repository (downloads once):

```bash
llama serve -hf ggml-org/Kev-4B-GGUF:Q4_K_M --host 127.0.0.1 --port 8080
```

Depending on your installed distribution, the executable may be `llama-server`, `llama`, or `./build/bin/llama-server`. If llama.cpp is older and returns **404** for `/v1/systemone`, update it. If you receive **501**, the loaded model/build does not support native decision inference.

On supported servers, `/v1/models` indicates native decision capability through `architecture.output_modalities: ["decisions"]`. Older servers may omit this metadata, so it is advisory rather than required.

To test the model server independently:

```bash
curl -s http://127.0.0.1:8080/health

curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state":"An industrial motor is hot, with high-frequency bearing impacts and periodic clicking.",
    "questions":{"fault":{"type":"choice","instructions":"What is the likely condition?",
      "criteria":{"bearing_wear":"Rolling element damage","lubrication":"Insufficient grease","normal":"No defect"}}}
  }'
```

That second command should return an `answers.fault` object with choice probabilities **if the endpoint works**. The example is a connectivity check, not a diagnostic benchmark.

### 2. Install Python dependencies

```bash
cd mechanical-failure-detective
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Smoke-test your Kev-4B server

```bash
python check_server.py --scenario bearing
```

This makes a **real** native System One request and prints probabilities. If it fails, fix the server before troubleshooting Streamlit.

### 4. Start the dashboard

```bash
streamlit run app.py --server.address 127.0.0.1
```

Open the URL Streamlit prints (typically **http://127.0.0.1:8501**).

The dashboard defaults to `http://127.0.0.1:8080` for llama.cpp. Change the port in the sidebar if needed. Leave the optional model alias blank when you serve one model; in router mode, set the exact model ID.

## A 30-minute interactive workshop

| Minutes | Activity | Learning objective |
| --- | --- | --- |
| 0–5 | Tour the bearing, vibration orders and simulator | Understand the mechanical evidence |
| 5–12 | Run normal, bearing wear, lubrication, imbalance | Explore `choice`, `score`, `noul` |
| 12–18 | Inject a high-load hot-but-quiet case | Expose conflicting evidence |
| 18–23 | Inject bad measurements and missing sensors | Encourage abstention and sensor validation |
| 23–28 | Run a 3–6 point temperature or vibration sweep | See decision stability/instability |
| 28–30 | Reveal ground truth and export log | Compare with a rule baseline |

### Suggested exercises

- Does a temperature increase by itself push the model toward lubrication failure?
- Will strong axial + 2× vibration overpower the operator's claim of “rotor imbalance”?
- Will the model choose **uncertain** with a missing vibration sensor?
- What happens to the choice probabilities when temperatures increase in small increments?
- Do the model's and baseline's failure modes differ?

The **Experiment log** records decisions and whether they match the injected scenario label. It is an **educational measurement only**, not a representative accuracy estimate.

## Architecture

```text
[Scenario library + editable sensor controls]
                  |
        [Synthetic time series]
                  |
           [State JSON]
              /       \
 [Python rules]       [requests POST /v1/systemone]
              |                     |
 [baseline point votes]     [llama.cpp + Kev-4B]
              \                     /
               [Streamlit dashboard]
                      |
       [Compare, perturb, export CSV]
```

Model requests contain an object-valued `state` and a `questions` object:

- `fault`: `choice` among bearing wear, lubrication, misalignment, imbalance, normal, and uncertain.
- `severity`: `score` using five ordered levels (0 through 4).
- `inspect`: `noul` estimating whether a technician should investigate.

For the sensitivity lab, only `fault` is requested per sample to reduce processing time.

## Safety, privacy and limitations

- This is a **simulation** with intentionally invented vibration indices; it does not analyze raw acoustic signals or assess real machine condition.
- Calibration, machine-specific engineering limits and ground-truth inspection evidence are **not provided**. Model probabilities **must not** be interpreted as probabilities of actual failures or used for equipment shutdown.
- Input validation flags are illustrative. Always use established engineering specifications, qualified technicians, and your organization's maintenance/safety procedures.
- Local HTTP is restricted by the client to loopback hosts (`127.0.0.1`, `localhost`, `::1`). No credentials, telemetry, or model requests leave the laptop as part of the Python demo.
- `Check server` is a reachability check; it does **not** guarantee that the model's `/v1/systemone` endpoint will return a valid response. Run one diagnostic to verify that.
- No physical motor, actuator, or external equipment is controlled.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests cover simulated scenarios, input validity, baseline rankings, HTTP request contract, mocked successful response parsing, malformed responses, and informative server errors. **They do not demonstrate live inference with your local Kev-4B; that requires running your own llama.cpp server.**

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `Connection refused` | Is llama.cpp running? Is the port correct? Start it before querying. |
| `404` | Upgrade llama.cpp; your build may predate `/v1/systemone`. |
| `501` | Confirm it loaded a **decision model**, not an ordinary chat model. |
| `400` | Check your model's supported decision API and update llama.cpp if necessary. |
| No decision metadata | Older `/v1/models` implementations may omit the field. Try a real inference call. |
| Slow on CPU | Use Kev-4B Q4_K_M, and keep the sensitivity sweep to 3 steps. |
| Old results after changing sliders | Intentionally marked stale; click Analyze again. |

### Upstream documentation

- [llama.cpp native System One API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api)
- [Kev-4B GGUF model card](https://huggingface.co/ggml-org/Kev-4B-GGUF)

Educational sample to adapt for internal engineering workshops. No implied certification for real diagnostic use.
