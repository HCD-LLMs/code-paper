# src/step_2_img/loop_temp_prompt.py
import os
from pathlib import Path
import time
import json
import importlib
from typing import Iterable, List, Dict, Any

import mlflow
from openai import OpenAI as RawOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def _import_prompt_modules(prototype: str, prompt_step: str = "step_2_prompt"):
    """
    Load dynamically the three prompt modules for the chosen prototype.
    Attempt to: src/<prompt_step>/<prototype>/{template,prompt_one_shot,prompt_few_shot}.py
    """
    base = f"src.{prompt_step}.{prototype}"
    zero_shot = importlib.import_module(f"{base}.template")
    one_shot  = importlib.import_module(f"{base}.prompt_one_shot")
    few_shot  = importlib.import_module(f"{base}.prompt_few_shot")
    return {
        "zero_shot": {"template": zero_shot.prompt_template, "text": zero_shot.prompt_text},
        "one_shot":  {"template": one_shot.prompt_template,  "text": one_shot.prompt_text},
        "few_shot":  {"template": few_shot.prompt_template,  "text": few_shot.prompt_text},
    }

# Resolve image paths
def _resolve_image_paths(prototype: str, data_root: str = "data") -> list[str]:
    base = Path(data_root) / prototype / "img"
    valid = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    return [str(p) for p in sorted(base.iterdir()) if p.suffix.lower() in valid]


def run(
    prototype: str = "rhyno_cyt",
    prompt_step: str = "step_2_prompt",
    experiment_name: str | None = None,
    models: Iterable[str] = ("llama3.2-vision:11b", "gemma3:12b", "qwen2.5vl:latest"),
    model_choice: int = 0,
    temperatures: Iterable[float] = tuple(i * 0.1 for i in range(11)),
    num_generations: int = 10,
    img_folder: str = "img",
    tracking_uri: str = "http://127.0.0.1:5000",
    openai_api_base: str = "http://localhost:11434/v1",
    openai_api_key: str = "ollama",
    extra_tags: Dict[str, Any] | None = None,
):
    """
    Entry point invoked from main. Example:
    python main.py --step step_2_img --param prototype=rhyno_cyt --param num_generations=5
    """
    # MLflow
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or f"Prompt_Comparison-{prototype}-img"
    mlflow.set_experiment(exp_name)
    mlflow.openai.autolog()

    # Load prompt dynamically
    prompts = _import_prompt_modules(prototype=prototype, prompt_step=prompt_step)
    # Load all images
    image_paths = _resolve_image_paths(prototype)
    if not image_paths:
        print(f"[WARN] nessuna immagine trovata in: data/{prototype}/img")


    
    models = list(models)
    chosen_model = models[model_choice]

    for prompt_name, prompt_data in prompts.items():
        
        tmpl = prompt_data["template"]
        if "image" not in getattr(tmpl, "input_variables", []):
            tmpl = PromptTemplate.from_template(prompt_data["text"])
            # ensure the text contains {image}

        for t in temperatures:
            chat_llm = ChatOpenAI(
                model=chosen_model,
                openai_api_base=openai_api_base,
                openai_api_key=openai_api_key,
                temperature=float(t),
            )
            chain = tmpl | chat_llm

            run_name = f"{chosen_model}_{prompt_name}_temp_{float(t):.1f}"
            with mlflow.start_run(run_name=run_name):
                # tag/parameters
                mlflow.set_tag("framework", "LangChain")
                mlflow.set_tag("prompt_type", prompt_name)
                mlflow.set_tag("prototype", prototype)
                if extra_tags:
                    for k, v in extra_tags.items():
                        mlflow.set_tag(k, v)

                mlflow.log_param("model", chosen_model)
                mlflow.log_param("temperature", float(t))
                mlflow.log_param("num_generations", int(num_generations))

                
                
                mlflow.log_artifacts(img_folder, artifact_path=f"input_images_{prompt_name}")

                # generazioni
                for gen in range(1, int(num_generations) + 1):
                    start = time.time()
                    answer = chain.invoke({"image": image_paths})
                    latency = time.time() - start

                    
                    output_text = getattr(answer, "content", str(answer))
                    mlflow.log_text(output_text, f"output_{prompt_name}_gen_{gen}.txt")
                    mlflow.log_metric(f"latency_gen_{gen}", latency)

            print(f"Run completed: model={chosen_model}, prompt={prompt_name}, temp={float(t):.1f}")
