"""Worker process of one environment. Serve mode: `main.py <env>`, JSON lines on stdin and on the original stdout.
Download mode: `main.py <env> download <stage>`. Every `run` has an `id` unique within the Worker process; a `cancel`
names that id. Standard library only."""
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


def drop_model(loaded):
    """Releases the loaded model."""
    loaded["stage"] = loaded["model"] = None
    release_memory()


def error_kind(error):
    if isinstance(error, InputError):
        return "input"
    if isinstance(error, Cancelled):
        return "cancelled"
    if type(error).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(error):
        return "oom"
    return "failed"


def run_stage(env, message, send, cancelled, loaded):
    """Runs one `run` message and sends its `done` or `error` message."""
    started = time.monotonic()
    run_id, stage = message["id"], message["stage"]
    ctx = Context(run_id, Path(message["dir"]), send, cancelled)
    try:
        if loaded["stage"] != stage:
            drop_model(loaded)
        module = stage_module(env, stage)
        if loaded["stage"] is None:
            if hasattr(module, "load"):
                ctx.progress(0.0, f"loading {stage}")
                loaded["model"] = module.load()
            loaded["stage"] = stage
        ctx.model = loaded["model"]
        result = module.run(ctx, **message["args"])
        send({"type": "done", "id": run_id, "result": result, "seconds": time.monotonic() - started})
    except Exception as error:
        kind = error_kind(error)
        if kind in ("oom", "failed"):
            traceback.print_exc()
        text = f"{type(error).__name__}: {error}" if kind == "failed" else str(error)
        send({"type": "error", "id": run_id, "kind": kind, "message": text})
        release_memory()
    finally:
        cancelled.discard(run_id)


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

    requests, cancelled, loaded = queue.Queue(), set(), {"stage": None, "model": None}
    threading.Thread(target=read_requests, args=(requests, cancelled), daemon=True).start()
    while (message := requests.get()) is not None:
        run_stage(env, message, send, cancelled, loaded)


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
