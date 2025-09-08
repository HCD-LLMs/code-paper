# main.py
import argparse
import importlib
from typing import Callable, Dict

PROMPT_STEPS = {"step_1_prompt", "step_2_prompt"}
LOOP_STEPS   = {"step_1_html", "step_2_img"}

MODE_TO_MODULE = {
    "zero-shot": "template",
    "one-shot":  "prompt_one_shot",
    "few-shot":  "prompt_few_shot",
}

def import_module(dotted: str):
    return importlib.import_module(dotted)

def get_entrypoint(mod) -> Callable:
    for name in ("run", "main", "build"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"{mod.__name__} must expose run()/main()/build().")

def auto_cast(val: str):
    v = val.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v) if "." not in v else float(v)
    except ValueError:
        return v

def parse_kv(params: list[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for p in params:
        k, sep, v = p.partition("=")
        if not sep:
            raise ValueError(f"Invalid --param '{p}', expected key=value")
        out[k.strip()] = auto_cast(v)
    return out

def main():
    ap = argparse.ArgumentParser("Unified runner")
    ap.add_argument("--step", required=True, choices=sorted(PROMPT_STEPS | LOOP_STEPS))
    ap.add_argument("--prototype", choices=["brainmed", "rhyno_cyt"])
    ap.add_argument("--mode", choices=list(MODE_TO_MODULE.keys()))
    ap.add_argument("--param", action="append", default=[], help="key=value (repeatable)")
    args = ap.parse_args()

    kwargs = parse_kv(args.param)

    if args.step in PROMPT_STEPS:
        if not args.prototype or not args.mode:
            ap.error("--prototype and --mode are required for prompt steps")
        module_name = MODE_TO_MODULE[args.mode]
        dotted = f"src.{args.step}.{args.prototype}.{module_name}"
    else:
        dotted = f"src.{args.step}.loop_temp_prompt"

    mod = import_module(dotted)
    entry = get_entrypoint(mod)
    entry(**kwargs)

if __name__ == "__main__":
    main()
