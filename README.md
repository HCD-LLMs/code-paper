# Prompt Engineering with Ollama + LangChain + MLflow

This guide explains how to configure a local environment to track experiments using **Ollama**, **LangChain**, and **MLflow**.  
It also describes the folder structure and how to run the provided Python scripts.

---

## Table of Contents

1. [Environment setup (venv)](#environment-setup-venv)  
2. [Installing dependencies](#installing-dependencies)  
3. [Starting the MLflow server](#starting-the-mlflow-server)  
4. [Running Ollama (models)](#running-ollama-models)  
5. [Folder structure](#folder-structure)  
6. [Running experiments](#running-experiments)  
7. [Notes](#notes)  

---

## Environment setup (venv)

### Windows

```powershell
cd C:\code-paper
python -m venv .venv
.venv\Scripts\activate
````

### macOS / Linux

```bash
cd ~/code-paper
python3 -m venv .venv
source .venv/bin/activate
```

---

## Installing dependencies

With the virtual environment active, install all required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Starting the MLflow server

Open another terminal (outside the virtual environment) and start MLflow:

### Windows

```powershell
mlflow server `
  --backend-store-uri file:./mlruns `
  --default-artifact-root file:./mlruns/artifacts `
  --host 127.0.0.1 --port 5000
```

### macOS / Linux

```bash
mlflow server \
  --backend-store-uri file:./mlruns \
  --default-artifact-root file:./mlruns/artifacts \
  --host 127.0.0.1 --port 5000
```

* **MLflow Dashboard**: open [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Running Ollama (models)

Make sure **Ollama** is installed and running:

```powershell
ollama run deepseek-r1:8b   # or another model (llama3, gemma3, qwen, etc.)
```

* The Ollama API will be available at: `http://localhost:11434`

---

## Folder structure

The project is organized as follows:

```
CODE-PAPER/
│
├── data/                         # Input data for prompts
│   ├── brainmed/
│   │   ├── html/                 # HTML files for BrainMed prototype
│   │   │   └── code_brainmed.html (example)
│   │   └── img/                  # Images for BrainMed prototype
│   │       └── ...
│   │
│   └── rhyno_cyt/
│       ├── html/                 # HTML files for Rhyno-Cyt prototype
│       │   └── code_rhyno.html
│       └── img/                  # Images for Rhyno-Cyt prototype
│           └── classified-cells-correct.png
│           └── classified-cells-wrong.png
│           └── dashboard.png
│
├── results/                      # Output folder (MLflow runs, artifacts, etc.)
│
├── src/                          # Source code
│   ├── step_1_html/
│   │   └── loop_temp_prompt.py   # Runs HTML-based experiments
│   ├── step_2_img/
│   │   └── loop_temp_prompt.py   # Runs image-based experiments
│   ├── step_1_prompt/            # Prompt templates for BrainMed & Rhyno-Cyt (HTML)
│   │   ├── brainmed/
│   │   │   ├── template.py
│   │   │   ├── prompt_one_shot.py
│   │   │   └── prompt_few_shot.py
│   │   └── rhyno_cyt/
│   │       └── ...
│   ├── step_2_prompt/            # Prompt templates for BrainMed & Rhyno-Cyt (Images)
│   │   ├── brainmed/
│   │   └── rhyno_cyt/
│   └── __init__.py
│
├── main.py                       # Unified entrypoint for all experiments
├── requirements.txt              # Dependencies
└── README.md                     # This guide
```

---

## Running experiments

All experiments are launched via **main.py** with parameters:

```bash
python main.py --step <step> [--prototype <name>] [--mode <prompt_mode>] [--param key=value]
```

### 1. Run HTML experiments (step 1)

```bash
python main.py --step step_1_html --param prototype=brainmed
python main.py --step step_1_html --param prototype=rhyno_cyt
```

* Loads HTML files from: `data/<prototype>/html/`

### 2. Run Image experiments (step 2)

```bash
python main.py --step step_2_img --param prototype=brainmed
python main.py --step step_2_img --param prototype=rhyno_cyt
```

* Loads image files from: `data/<prototype>/img/`

### 3. Run Prompt-only experiments (text-based)

```bash
# Zero-shot, one-shot, few-shot for BrainMed
python main.py --step step_1_prompt --prototype brainmed --mode zero-shot
python main.py --step step_1_prompt --prototype brainmed --mode one-shot
python main.py --step step_1_prompt --prototype brainmed --mode few-shot

# Rhyno-Cyt (same for step_2_prompt)
python main.py --step step_2_prompt --prototype rhyno_cyt --mode zero-shot
```

### 4. Extra parameters

You can pass additional parameters with `--param key=value`, for example:

```bash
python main.py --step step_2_img --param prototype=brainmed --param num_generations=5 --param model_choice=0 --param temperature=0.3
```



