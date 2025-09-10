# Can the Tasks of your Usability Test be Scripted by an LLM?  _A Case Study in the Medical Domain_  

This repository contains all the material related to the paper **"Can the Tasks of your Usability Test be Scripted by an LLM? A Case Study in the Medical Domain"**.  
It is organized to provide a clear navigation of prompts, prototypes, source code, and results.  



## 📂 Repository Structure

### Root files
- **.gitignore** → Standard Git configuration for excluding temporary/local files.  
- **HCD and LLMs [Anonymous Submission].pdf** → Anonymous submission of the paper.   
- **README.md** → This file, describing the repository content.  
- **requirements.txt** → Python dependencies required to run the source code.  



### 📂 Prompt_iterations
This folder contains the evolution of the prompts designed to generate tasks with LLMs.  

- **Prompt_v1.txt – Prompt_v3.txt** → Early iterations of prompt design.  
- **Prompt_v4/**  
  - **Anonymous_prompt_v4/** → Prompts adapted for the _Anonymous_ prototype.  
  - **BrainMed_prompt_v4/** → Prompts adapted for the _BrainMed_ prototype.  
- **Prompt_v5/**  
  - **Anonymous_prompt_v5/** → Refined prompts for the Anonymous prototype.  
  - **BrainMed_prompt_v5/** → Refined prompts for the BrainMed prototype.  



### 📂 Prototypes
Contains the **two medical prototypes** used as case studies in the experiments. Each prototype includes both HTML code and images.  

- **anonymous/**  
  - **html/** → HTML implementation of the Anonymous prototype pages.  
  - **img/** → Images supporting the prototype (UI screenshots, visual assets).  

- **brainmed/**  
  - **html/** → HTML implementation of the BrainMed prototype pages.  
  - **img/** → Images supporting the prototype.  


### 📂 Results
Contains the results of the **user study** and the **LLM-based task generation analysis**.  

- **User_study_results.xlsx** → Raw results from the questionnaire.  
- **gemma/**
    - `analysis_task.xlsx` → Analysis of results for the _Gemma_ model.  
    - `final_task_generated_anonymous.xlsx` → Tasks generated for the Anonymous prototype using Gemma.  
    - `final_task_generated_brainmed.xlsx` → Tasks generated for the BrainMed prototype using Gemma.
- **llama/** 
    - `analysis_task.xlsx` → Analysis of results for the _Llama_ model.  
    - `final_tasks_anonymous.xlsx` → Tasks generated for the Anonymous prototype using Llama.  
    - `final_tasks_brainmed.xlsx` → Tasks generated for the BrainMed prototype using Llama.
- **qwen/** 
    - `final_tasks_analysis_anonymous.xlsx` → Tasks generated and analysis for the Anonymous prototype using Qwen.  
    - `final_tasks_analysis_brainmed.xlsx` → Tasks generated and analysis for the BrainMed prototype using Qwen.  


### 📂 Source Code
Contains the Python code used to run the LLM experiments and manage prompts.  

- **Source_code_Anonymous/**  
  - `loop_temp_prompt.py` → Main script to run prompt iterations for the Anonymous prototype.  
  - `prompt_few_shot.py` → Few-shot prompt template.
  - `prompt_one_shot.py` → One-shot prompt template.  
  - `prompt_zero_shot.py` → Zero-shot prompt template.

- **Source_code_BrainMed/**  
  - `loop_temp_prompt.py` → Main script to run prompt iterations for the BrainMed prototype.  
  - `prompt_few_shot.py` → Few-shot prompt template
  - `prompt_one_shot.py` → One-shot prompt template.  
  - `prompt_zero_shot.py` → Zero-shot prompt template.  



## ⚙️ How to Use

### 1. Environment Setup (venv)

#### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
````

#### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install Dependencies

With the virtual environment activated, install all required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Start MLflow Server

Open another terminal (outside the virtual environment) and start the local MLflow server:

#### Windows

```powershell
mlflow server `
  --backend-store-uri file:./mlruns `
  --default-artifact-root file:./mlruns/artifacts `
  --host 127.0.0.1 --port 5000
```

* **Dashboard**: open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to access MLflow UI.

---

### 4. Start Ollama (Run Model)

Make sure you have [Ollama](https://ollama.com) installed on your system.
Run the desired model, for example:

```bash
ollama run llama3.2-vision:11b # or "gemma3:12b",  qwen25vl:latest"
```

* The model server will run at `http://localhost:11434`.

Based on the model you choose, adjust in the script `loop_temp_prompt.py` the variable model_choice .

---

### 5. Run Python Scripts

With both **MLflow** and **Ollama** running and the virtual environment activated, execute one of the experiment scripts. For example:

```bash
python Source_Code/Source_code_BrainMed/loop_temp_prompt.py
```

or

```bash
python Source_Code/Source_code_Anonymous/loop_temp_prompt.py
```


* MLflow will automatically log runs, including **prompts, responses, parameters, and artifacts**.


