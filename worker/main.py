"""Worker process of one environment. Serve mode: `main.py <env>`, JSON lines on stdin and on the original stdout.
Download mode: `main.py <env> download <stage>`. Every `run` has an `id` unique within the Worker process; a `cancel`
names that id. Standard library only.

A stage is the module `<root>/stages/<env>/<stage>.py` with the function `run(ctx, **args)`. A stage with a model also
has `load()` (returns the model, which `run` finds as `ctx.model`), `download()` and `KEEP_LOADED` (True or False); a
stage without a model has none of them and `ctx.model` is None. A model is loaded before the first run of its stage
and stays loaded until another model is loaded: loading the model of a stage with KEEP_LOADED False first drops every
loaded model of a stage with KEEP_LOADED False; the models of stages with KEEP_LOADED True stay loaded for the life of
the process."""
import gc
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

from context import MODELS, Cancelled, Context, InputError

ROOT = Path(__file__).resolve().parents[1]


def setup(offline):
    """Sets the environment variables and sys.path that stage imports need."""
    os.environ["HF_HOME"] = str(MODELS / "hf")
    os.environ["TORCH_HOME"] = str(MODELS / "torch")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "shared"))


def stage_module(env, stage):
    """Returns the module of `<root>/stages/<env>/<stage>.py`, loaded once per process as `stage_<stage>` in
    sys.modules."""
    name = f"stage_{stage}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "stages" / env / f"{stage}.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def release_memory():
    """Collects garbage; when torch is imported, also empties the CUDA cache and releases the cached pinned host
    memory."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()


def load_model(env, stage, module, models, ctx):
    """Loads the model of `stage` into `models` (models by stage name), first dropping the models of the stages with
    KEEP_LOADED False when the stage itself has KEEP_LOADED False."""
    if not module.KEEP_LOADED:
        dropped = [name for name in models if not stage_module(env, name).KEEP_LOADED]
        for name in dropped:
            del models[name]
        if dropped:
            release_memory()
    ctx.progress(0.0, f"loading {stage}")
    models[stage] = module.load()


def error_kind(error):
    if isinstance(error, InputError):
        return "input"
    if isinstance(error, Cancelled):
        return "cancelled"
    if type(error).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(error):
        return "oom"
    return "failed"


def run_stage(env, message, send, cancelled, models):
    """Runs one `run` message and sends its `done` or `error` message. `models` holds the loaded models by stage
    name."""
    started = time.monotonic()
    run_id, stage = message["id"], message["stage"]
    ctx = Context(run_id, Path(message["dir"]), send, cancelled)
    failed = False
    try:
        module = stage_module(env, stage)
        if hasattr(module, "load") and stage not in models:
            load_model(env, stage, module, models, ctx)
        ctx.model = models.get(stage)
        result = module.run(ctx, **message["args"])
        send({"type": "done", "id": run_id, "result": result, "seconds": time.monotonic() - started})
    except Exception as error:
        failed = True
        kind = error_kind(error)
        if kind in ("oom", "failed"):
            traceback.print_exc()
        text = f"{type(error).__name__}: {error}" if kind == "failed" else str(error)
        send({"type": "error", "id": run_id, "kind": kind, "message": text})
    finally:
        cancelled.discard(run_id)
    if failed:
        release_memory()


def read_requests(requests, cancelled):
    """Reads stdin: `run` messages go to `requests`, `cancel` messages into `cancelled`; None marks EOF."""
    for line in sys.stdin:
        message = json.loads(line)
        if message["type"] == "cancel":
            cancelled.add(message["id"])
        else:
            requests.put(message)
    requests.put(None)


def serve(env):
    setup(offline=True)
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(message):
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    requests, cancelled, models = queue.Queue(), set(), {}
    threading.Thread(target=read_requests, args=(requests, cancelled), daemon=True).start()
    while (message := requests.get()) is not None:
        run_stage(env, message, send, cancelled, models)


def download(env, stage):
    setup(offline=False)
    module = stage_module(env, stage)
    if hasattr(module, "download"):
        module.download()


if __name__ == "__main__":
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    if len(sys.argv) == 4:
        download(sys.argv[1], sys.argv[3])
    else:
        serve(sys.argv[1])
