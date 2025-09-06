import json, time
import mlflow
from openai import OpenAI as RawOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from template import prompt_template, prompt_text

# MLflow setup (locale)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Prompt_Comparison")
mlflow.openai.autolog()  # traccia automaticamente le chiamate

# (Opzionale) Client "raw" Ollama/OpenAI per embedding o moderazione
raw_client = RawOpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)

models = ["llama3.1:8b", "gemma3:4b", "deepseek-r1:8b", "phi4:14b", "qwen2.5", "qwen3:8b"]
model_choice = 5
html_files = ["code_rhyno.html"]

# PromptTemplate e pipeline
prompt_template.input_variables = ["html_snippet"]

# Funzione per caricare HTML
def load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Loop sulle temperature da 0.0 a 1.0 con step 0.1
for t in [i * 0.1 for i in range(11)]:
    # Inizializza LLM con la temperatura corrente
    chat_llm = ChatOpenAI(
        model=models[model_choice],
        openai_api_base="http://localhost:11434/v1",
        openai_api_key="ollama",
        temperature=t
    )
    chain = prompt_template | chat_llm

    run_name = f"LangChain_Ollama_Run_temp_{t:.1f}"
    with mlflow.start_run(run_name=run_name):
        # Parametri di logging
        mlflow.set_tag("framework", "LangChain")
        mlflow.log_param("model", models[model_choice])
        mlflow.log_param("temperature", t)
        mlflow.log_param("num_files", len(html_files))

        results = {}
        for idx, file in enumerate(html_files, 1):
            html = load_html(file)
            # Log del prompt e del file HTML
            mlflow.log_text(html, f"code_rhino_cyt{idx}.html")
            mlflow.log_text(prompt_text, f"prompt{idx}.html")

            start = time.time()
            answer = chain.invoke({"html_snippet": html})
            latency = time.time() - start

            results[file] = answer
            mlflow.log_text(answer.text(), f"output_{idx}.txt")
            mlflow.log_metric(f"latency_{idx}", latency)

    print(f"Run completed for temperature = {t:.1f}")
