import os
import json, time
import mlflow
from openai import OpenAI as RawOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Import con alias per non sovrascrivere i nomi
import template as zero_shot
import prompt_one_shot as one_shot
import prompt_few_shot as few_shot

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Prompt_Comparison-rhyno-cyt-img")
mlflow.openai.autolog()

# (Opzionale) Raw client per embedding o moderazione
raw_client = RawOpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)

models = ["llama3.2-vision:11b", "gemma3:12b", "qwen2.5vl:latest"]
model_choice = 0
num_generations = 10

# Raggruppa i tuoi prompt in un dict:
prompt_variants = {
    "zero_shot": {
        "template": zero_shot.prompt_template,
        "text": zero_shot.prompt_text
    },
    "one_shot": {
        "template": one_shot.prompt_template,
        "text": one_shot.prompt_text
    },
    "few_shot": {
        "template": few_shot.prompt_template,
        "text": few_shot.prompt_text
    }
}

# Funzione per caricare HTML
# Funzione per caricare HTML
def load_image_paths(folder: str = "img") -> list[str]:
    valid_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    return [
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if os.path.splitext(fname.lower())[1] in valid_exts
    ]

# Loop su modello, tipo di prompt e temperature
image_paths = load_image_paths("img")
for prompt_name, prompt_data in prompt_variants.items():
    # Assicuriamoci che il template prenda in input i path delle immagini
    prompt_data["template"].input_variables = ["image"]

    for t in [i * 0.1 for i in range(11)]:
        chat_llm = ChatOpenAI(
            model=models[model_choice],
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            temperature=t
        )
        chain = prompt_data["template"] | chat_llm

        run_name = f"{models[model_choice]}_{prompt_name}_temp_{t:.1f}"
        with mlflow.start_run(run_name=run_name):
            # Tag e parametri
            mlflow.set_tag("framework", "LangChain")
            mlflow.set_tag("prompt_type", prompt_name)
            mlflow.log_param("model", models[model_choice])
            mlflow.log_param("temperature", t)
            mlflow.log_param("num_generations", num_generations)

            # Log dei path immagini come artifact
            # Registra l'intera cartella 'img' (tutte le immagini) per riferimento
            mlflow.log_artifacts("img", artifact_path=f"input_images_{prompt_name}")

            for gen in range(1, num_generations + 1):
                start = time.time()
                # Passiamo la lista di path al modello
                answer = chain.invoke({"image": image_paths})
                latency = time.time() - start

                mlflow.log_text(
                    answer.text(),
                    f"output_{prompt_name}_gen_{gen}.txt",
                )
                mlflow.log_metric(f"latency_gen_{gen}", latency)

        print(f"Run completed: model={models[model_choice]}, prompt={prompt_name}, temp={t:.1f}")