


# AI Agent Automation (Medical Diagnostics)

A multi-agent system that analyzes patient medical reports using specialized AI “experts” (Cardiologist, Psychologist, Pulmonologist) and synthesizes their findings into a final, concise recommendation. Includes a Flask web API/UI and an evaluator to benchmark model performance on a large synthetic dataset.

> Note: This README reflects the structure and code found in this project (CLI app, Flask API, Agents in `Utils/Agents.py`, and evaluator in `Utils/evaluate_agent.py`). Verify model/provider settings and environment variables before running.

---
## The code was tested on google colab enabled with T4GPU accelerator,  GPUs with VRAM of <16GB will not be able to run the code 
## Contents
- Overview
- Features
- Architecture
- Project Structure
- Quickstart
- Configuration (.env)
- Running the CLI
- Running the Web App (Flask)
- API Reference
- Evaluation (Utils/evaluate_agent.py)
- Troubleshooting
- Development Notes
- License and Disclaimer

---

## Overview
This project orchestrates multiple domain-specific AI agents to analyze a single medical report:
- Cardiologist: cardiac-focused analysis
- Psychologist: mental health perspective
- Pulmonologist: respiratory assessment
- Multidisciplinary Team: synthesizes specialist outputs into a final diagnosis/recommendation

The system exposes:
- A command-line interface (CLI) workflow
- A Flask web server with REST endpoints and a basic HTML UI
- An evaluation script to measure accuracy on a large synthetic dataset

---

## Features
- Multi-agent orchestration with role-specific prompts
- Concurrent execution for faster end-to-end analysis
- Flask API and simple HTML UI
- Configurable model backends (code indicates Groq in evaluator; Gemini/LC chainable in agents if configured)
- Large-scale evaluation (dataset up to 10,000 items) with practical rate-limit controls
- Windows-friendly examples and path handling tips

---

## Architecture
- Utils/Agents.py
  - Base Agent with role-specific prompt templates
  - Specialized agents: Cardiologist, Psychologist, Pulmonologist
  - MultidisciplinaryTeam agent that merges individual reports
  - Model backend: ensure you configure a provider and API key (see Configuration)
- flask_app.py
  - REST API
  - HTML rendering via templates/index.html (create if missing)
  - Concurrent agent execution and final synthesis
- Utils/evaluate_agent.py
  - Synthetic dataset generation (arithmetic, capitals, authors, science)
  - Strict and type-aware scoring to avoid inflated accuracy
  - Environment-driven caps to avoid rate limits
  - Saves detailed results JSON

---

## Project Structure
```
├─ Main.py                              # CLI (interactive) driver (WIP)
├─ flask_app.py                         # Flask web server and REST API
├─ Utils/
│  ├─ Agents.py                         # Agent classes and orchestration
│  └─ evaluate_agent.py                 # Model evaluation on large synthetic set
├─ Medical Reports/
│  └─ Medical Report - Michael Johnson - Panic Attack Disorder.txt  # sample
├─ templates/
│  └─ index.html                        # Web UI (create if missing)
├─ Results/
│  └─ eval_results.json                 # Evaluation output (created at runtime)
├─ requirements.txt                     # Dependencies (update/install as needed)
├─ .env                                 # Environment variables (you create)
└─ apikey.env                           # Optional alt env file (you create)
```

---

## Quickstart

1) Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```powershell
pip install -r requirements.txt
```

3) Configure your API keys (choose provider(s) you actually use):

```
- For Groq (used by evaluator as written):
```powershell
setx GROQ_API_KEY "YOUR_ACTUAL_KEY"
# or session-only:
$env:GROQ_API_KEY="YOUR_ACTUAL_KEY"
```

4) Important (GPU/torch): If you need CUDA builds, install per your CUDA version:
```powershell
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade transformers accelerate bitsandbytes
```
### Installing the 4) point is very important for running this model.
---

## Configuration (.env)

Create a `.env` in repo root (or use environment variables directly):

```
# Model/provider keys

GROQ_API_KEY=your_groq_key_here

# Evaluator controls (defaults shown)
EVAL_MODEL_ID=openai/gpt-oss-120b
EVAL_DATASET_SIZE=10000
EVAL_MAX_API_CALLS=300
EVAL_RANDOM_SEED=42
EVAL_OUTPUT=./Results/eval_results.json
```

Notes:
- `Utils/evaluate_agent.py` uses Groq in the provided code; set `GROQ_API_KEY` and `EVAL_MODEL_ID` appropriately.
- If you maintain keys in `apikey.env`, load it at runtime via `dotenv`.

---

## Running the CLI

The CLI reads a medical report and prints:
- Individual specialist outputs
- A synthesized final diagnosis

Run:
```powershell
.\.venv\Scripts\python.exe .\Main.py
```

Paste your report when prompted. Type END on a blank line to finish input.

Tip (Windows path): If you load a report from file in Python:
```python
with open(r"Medical Reports\Medical Report - Michael Johnson - Panic Attack Disorder.txt", "r", encoding="utf-8") as f:
    text = f.read()
```
Use a raw string (prefix r"...") or double backslashes to avoid invalid escape sequences.

---

## Running the Web App (Flask)

Start the server:
```powershell
.\.venv\Scripts\python.exe .\flask_app.py
```

Open in browser:
```
http://localhost:5000
```

If `templates/index.html` is missing, create a minimal page that POSTs to `/api/analyze`.

---

## API Reference

Base URL: `http://localhost:5000`

- GET `/`
  - Renders `templates/index.html`

- POST `/api/analyze`
  - Body (application/json):
    ```json
    { "medical_report": "Patient reports chest pain and palpitations..." }
    ```
  - Response (200):
    ```json
    {
      "success": true,
      "results": {
        "cardiologist": "...",
        "psychologist": "...",
        "pulmonologist": "...",
        "final_diagnosis": "..."
      },
      "errors": null
    }
    ```
  - If an agent fails, its error is returned under `"errors"` and the system still attempts synthesis with available reports.

- GET `/api/sample-report`
  - Returns the sample report from `Medical Reports/Medical Report - Michael Johnson - Panic Attack Disorder.txt` if present; otherwise a built-in fallback text is returned.

---

## Evaluation (Utils/evaluate_agent.py)

The evaluator generates a large, mixed dataset (up to 10,000 items) and scores model outputs using type-appropriate metrics. It also caps API calls to avoid rate limits.

Environment variables:
- `EVAL_DATASET_SIZE` (default 10000): total synthetic items to create
- `EVAL_MAX_API_CALLS` (default 300): maximum items actually sent to the model
- `EVAL_RANDOM_SEED` (default 42): reproducibility
- `EVAL_MODEL_ID` (default openai/gpt-oss-120b): Groq OSS model id as coded
- `GROQ_API_KEY`: required for Groq client
- `EVAL_OUTPUT` (default ./Results/eval_results.json): save path for results

Run:
```powershell
$env:EVAL_DATASET_SIZE="10000"
$env:EVAL_MAX_API_CALLS="500"
$env:EVAL_OUTPUT="Results/eval_results.json"
$env:EVAL_MODEL_ID="openai/gpt-oss-120b"
.\.venv\Scripts\python.exe .\Utils\evaluate_agent.py
```

Output:
- Prints accuracy on the evaluated subset
- Writes detailed per-item results and config to `Results/eval_results.json`

Scoring strategy (designed for “good but not overfitting” accuracy):
- Arithmetic/Capitals: exact-ish canonical token checks
- Authors: accept canonical variants (e.g., “Shakespeare” or “William Shakespeare”)
- Science: keyword token-F1 with a threshold (e.g., 0.55) to credit meaningful coverage while penalizing generic fluff

To switch evaluator to Gemini:
- Replace Groq client/model code with `langchain-google-genai` Chat model initialization
- Or accept Gemini prompts via LangChain abstractions (ensure `GOOGLE_API_KEY` is set)

---

## Troubleshooting

- Windows path escapes:
  - Use raw strings or double slashes:
    ```python
    open(r"Medical Reports\file.txt")  # ok
    open("Medical Reports\\file.txt")  # ok
    ```


- Groq auth:
  - Set `GROQ_API_KEY`. Verify `EVAL_MODEL_ID` is a supported ID for your account.

- CUDA/Torch:
  - Match your CUDA version in the torch install command or use CPU-only wheels.

- Rate limits:
  - Tune `EVAL_MAX_API_CALLS`. Consider lighter models (e.g., Gemini “flash” variants) for large runs.

- CORS/UI:
  - `flask_app.py` enables CORS. If browsers block calls, confirm correct origin and headers.

---

## Development Notes

- Concurrency:
  - Agents run concurrently using `ThreadPoolExecutor` in both the Flask API and (intended) CLI for responsiveness.

- Templates/UI:
  - Ensure `templates/index.html` exists if you use the web UI. A simple form that POSTs to `/api/analyze` is sufficient.

- Logging/Results:
  - Evaluator writes JSON into `Results/`. Create the folder if missing.

- Extending agents:
  - Add new specialist classes in `Utils/Agents.py`.
  - Update the MultidisciplinaryTeam prompt template to incorporate new roles.

---

## License and Disclaimer

- License: See `LICENSE` in the repository (if present).
- Medical Disclaimer: This software is for research and educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

```
<!-- filepath: README.md -->
# AI Agent Automation (Medical Diagnostics)

A multi-agent system that analyzes patient medical reports using specialized AI “experts” (Cardiologist, Psychologist, Pulmonologist) and synthesizes their findings into a final, concise recommendation. Includes a Flask web API/UI and an evaluator to benchmark model performance on a large synthetic dataset.

> Note: This README reflects the structure and code found in this project (CLI app, Flask API, Agents in `Utils/Agents.py`, and evaluator in `Utils/evaluate_agent.py`). Verify model/provider settings and environment variables before running.

```


## Name: Divyansh Gupta
## University : Indian Institute Of Technology, Patna
## Mechanical Branch
