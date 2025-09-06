# mlflow_langchain.py

import json, time
import mlflow
from openai import OpenAI as RawOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from template import prompt_template
from template import prompt_text

# MLflow setup (locale)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Prompt_Comparison")
mlflow.openai.autolog()  # traccia automaticamente le chiamate :contentReference[oaicite:5]{index=5}

temp = 0.3

# (Opzionale) Client "raw" Ollama/OpenAI per embedding o moderazione
raw_client = RawOpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)

models = ["llama3.1:8b", "gemma3:4b", "deepseek-r1:8b", "phi4:14b", "qwen2.5-coder", "qwen2.5", "qwen3:8b"]
model_choice = 3

# LLM via langchain-openai
chat_llm = ChatOpenAI(
    model=models[model_choice],
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
    temperature = temp
)

# PromptTemplate e pipeline
prompt_template.input_variables = ["html_snippet"]

# Composizione prompt→LLM con pipe (crea internamente RunnableSequence)
chain = prompt_template | chat_llm

def load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

html_files = ["code_rhyno.html"]

with mlflow.start_run(run_name="LangChain_Ollama_Run"):
    mlflow.set_tag("framework", "LangChain")
    mlflow.log_param("model", models[model_choice])
    mlflow.log_param("num_files", len(html_files))

    results = {}
    for idx, file in enumerate(html_files, 1):
        html = load_html(file)
        mlflow.log_param("temperature", temp)
        mlflow.log_text(html, f"code_rhino_cyt{idx}.html")
        mlflow.log_text(prompt_text, f"prompt{idx}.html")

        start = time.time()
        answer = chain.invoke({"html_snippet": html})
        latency = time.time() - start

        results[file] = answer
        mlflow.log_text(answer.text(), f"output_{idx}.txt")
        mlflow.log_metric(f"latency_{idx}", latency)
        