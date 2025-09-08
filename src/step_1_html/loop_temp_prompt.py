import time
import glob
import importlib
from pathlib import Path
from typing import Iterable, Dict, Any


import mlflow
from openai import OpenAI as RawOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def _import_prompt_modules(prototype: str, prompt_step: str = "step_1_prompt"):
    """
    Carica dinamicamente i moduli dei prompt per il prototipo scelto.
    Atteso: src/<prompt_step>/<prototype>/{template,prompt_one_shot,prompt_few_shot}.py
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

# --- ADD: risoluzione percorsi HTML in base a prototype
def _resolve_html_paths(prototype: str, data_root: str = "data") -> list[Path]:
    base = Path(data_root) / prototype / "html"
    # Prendi tutti gli .html presenti (ordinati)
    return sorted(base.glob("*.html"))



def _load_html_by_glob(pattern: str) -> str:
    """Concatena tutti i file che matchano il glob nell'ordine alfabetico."""
    parts = []
    for p in sorted(glob.glob(pattern)):
        parts.append(Path(p).read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def run(
    prototype: str = "brainmed",
    prompt_step: str = "step_1_prompt",
    experiment_name: str | None = None,
    # modelli & sampling
    models: Iterable[str] = (
        "llama3.1:8b", "gemma3:12b", "gemma3:4b",
        "deepseek-r1:8b", "phi4:14b", "mistral:7b",
        "qwen2.5", "qwen3:8b"
    ),
    model_choice: int = 0,
    temperatures: Iterable[float] = tuple(i * 0.1 for i in range(11)),
    num_generations: int = 10,
    # connessioni
    tracking_uri: str = "http://127.0.0.1:5000",
    openai_api_base: str = "http://localhost:11434/v1",
    openai_api_key: str = "ollama",
    extra_tags: Dict[str, Any] | None = None,
):
    """
    Esempio CLI:
    python main.py --step step_1_html \
      --param prototype=brainmed \
      --param html_glob=code_brainmed_*.html \
      --param num_generations=5
    """
    # --- MLflow
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or f"Prompt_Comparison_{prototype}_html"
    mlflow.set_experiment(exp_name)
    mlflow.openai.autolog()

    # --- (opzionale) client raw
    _ = RawOpenAI(base_url=openai_api_base, api_key=openai_api_key)

    # --- prompt dinamici
    prompts = _import_prompt_modules(prototype=prototype, prompt_step=prompt_step)

    # --- carica tutti gli HTML
    # --- REPLACE: caricamento HTML
    html_paths = _resolve_html_paths(prototype)
    all_html = "\n\n".join(p.read_text(encoding="utf-8") for p in html_paths)
    if not all_html.strip():
        print(f"[WARN] nessun file HTML trovato in: data/{prototype}/html")


    models = list(models)
    chosen_model = models[model_choice]

    for prompt_name, prompt_data in prompts.items():
        # assicura che il template accetti {html_snippet}; se no ricrealo dal testo
        tmpl = prompt_data["template"]
        if "html_snippet" not in getattr(tmpl, "input_variables", []):
            tmpl = PromptTemplate.from_template(prompt_data["text"])
            # NB: il testo deve contenere {html_snippet}

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
                # tag/parametri
                mlflow.set_tag("framework", "LangChain")
                mlflow.set_tag("prompt_type", prompt_name)
                mlflow.set_tag("prototype", prototype)
                if extra_tags:
                    for k, v in extra_tags.items():
                        mlflow.set_tag(k, v)

                mlflow.log_param("model", chosen_model)
                mlflow.log_param("temperature", float(t))
                mlflow.log_param("num_generations", int(num_generations))

                # log degli input (HTML + testo prompt)
                mlflow.log_text(all_html,             f"input_html_{prompt_name}.html")
                mlflow.log_text(prompt_data["text"],  f"prompt_{prompt_name}.txt")

                # generazioni
                for gen in range(1, int(num_generations) + 1):
                    start = time.time()
                    answer = chain.invoke({"html_snippet": all_html})
                    latency = time.time() - start

                    output_text = getattr(answer, "content", str(answer))
                    mlflow.log_text(output_text, f"output_{prompt_name}_gen_{gen}.txt")
                    mlflow.log_metric(f"latency_gen_{gen}", latency)

            print(f"Run completed: model={chosen_model}, prompt={prompt_name}, temp={float(t):.1f}")
