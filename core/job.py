"""Job: the base class of the jobs in `<root>/jobs/`, the context their `run` gets, and the description of a job.

A job is the module `jobs/<name>.py` with the file `jobs/<name>.yaml` next to it. The module defines one subclass of
`Job`; the yaml file holds the keys
- `description`: the job description, at most one paragraph;
- `params`: one line per field of `Params`;
- `inputs`: one line per entry of `inputs`;
- `outputs`: one line per output file, named by its file name."""
import ast
import importlib
import inspect
import textwrap
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core.layout import ROOT


@dataclass(frozen=True)
class Input:
    """A named file input of a job: `kind` is the kind of the files, `minimum` and `maximum` the number of files sent."""
    kind: str
    minimum: int = 1
    maximum: int = 1


class BaseParams(BaseModel):
    """The base of the `Params` of a job: the names of the fields are the names of the params, and no other name is
    accepted."""
    model_config = ConfigDict(extra="forbid")


# The type of the `seed` param of a job with a seeded stage. The Api replaces "random" in the params with an integer
# when it creates the job; the Runner and the jobs see an integer.
Seed = Annotated[int, Field(ge=0, le=2**31 - 1)] | Literal["random"]


class Job:
    """The base of a job. A subclass defines
    - `Params`: a subclass of `BaseParams` with the type, limits and enum of every param, without default values;
    - `inputs`: {input name: Input}; a job without file inputs keeps the empty dict;
    - `async run(self, ctx)`: a linear sequence of `await ctx.local(stage, ...)` and `await ctx.cloud(app, ...)` calls
      with the literal name of the stage or app, then the assembly of the outputs. A condition on the params may skip
      a call; it never puts one call in place of another. A cloud call is the first call of a job."""

    inputs = {}

    @classmethod
    def stages(cls):
        """The calls of `run`, in source order, including those behind a condition, as [("local", stage) or ("cloud",
        app)]."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls.run)))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx"
                 and node.func.attr in ("local", "cloud")]
        calls.sort(key=lambda node: (node.lineno, node.col_offset))
        return [(node.func.attr, ast.literal_eval(node.args[0])) for node in calls]


class Context:
    """Passed to the `run` of a job, one per item of the job.

    - `params`: the validated `Params`; `inputs`: {input name: [file paths]}, a list for every input of the job;
    - `item`: the number of the item; `seed`: the `seed` param plus `item`, passed to every seeded stage; a job without
      a `seed` param has no `seed`;
    - `dir`: the folder of the item, below the job folder; every stage call gets its own folder in it."""

    def __init__(self, store, local_queue, cloud_queue, job, params, inputs, item, stage_count, report):
        """`stage_count` is the number of calls in `Job.stages()`. `report(item, fraction, message)` is a synchronous
        callback that receives the progress of the item, `fraction` in 0..1."""
        self.params, self.inputs, self.item = params, inputs, item
        self.dir = store.job_dir(job["job_id"]) / f"item_{item}"
        self.dir.mkdir()
        self._store, self._local, self._cloud = store, local_queue, cloud_queue
        self._job_id, self._priority = job["job_id"], job["priority"]
        self._charge = partial(store.add_work, job["job_id"])
        self._total, self._report = stage_count, report
        self._calls = 0

    @property
    def seed(self):
        return self.params.seed + self.item

    def file(self, name):
        """The path of the first file of the input `name`, None when no file was sent."""
        return next(iter(self.inputs[name]), None)

    async def local(self, stage, **args):
        """Runs the local stage `stage` with the keyword arguments `args` of its `run`, which are JSON-able, and returns
        the result dict of its `run`."""
        directory, progress = self._begin(stage)
        return await self._local.call(stage, args, directory, self._priority, progress, self._charge, self._job_id)

    async def cloud(self, app, files, params):
        """Runs the Modal app `app`; `files` is {argument name: path of the file}, `params` the params dict of the app.
        Returns its result dict: every file is written to its key in the folder of the call and replaced by its
        path."""
        directory, progress = self._begin(app)
        return await self._cloud.call(app, files, params, directory, self._priority, progress, self._charge,
                                      self._job_id)

    def output(self, path, kind):
        """Registers the existing file `path` below the job folder as an output of the item; `kind` is `image`, `mesh`
        or `audio`. Outputs are listed in the order they are registered."""
        self._store.add_output(self._job_id, self.item, Path(path), kind)

    def _begin(self, name):
        """Creates the folder of the next call and returns it with the progress callback of the call, which reports
        the item's progress when the call sends a message."""
        number, self._calls = self._calls, self._calls + 1
        directory = self.dir / f"{number:02d}_{name}"
        directory.mkdir()

        def progress(fraction, message):
            self._report(self.item, (number + fraction) / self._total, f"{name}: {message}")

        return directory, progress


def job_names():
    """The names of the jobs: the modules of `<root>/jobs/`, sorted."""
    return sorted(path.stem for path in (ROOT / "jobs").glob("*.py"))


def load_job(name):
    """The `Job` subclass defined in the module `jobs/<name>.py`."""
    module = importlib.import_module(f"jobs.{name}")
    return next(value for value in vars(module).values()
                if isinstance(value, type) and issubclass(value, Job) and value.__module__ == module.__name__)


def describe(job_class):
    """The description of a job as a dict:
    - `description`: the text of the yaml file;
    - `params`: the JSON schema of `Params`, each property with the description of the yaml file;
    - `inputs`: {input name: {"kind", "min", "max", "description"}};
    - `outputs`: {output file name: description}."""
    text = yaml.safe_load(Path(inspect.getfile(job_class)).with_suffix(".yaml").read_text(encoding="utf-8"))
    schema = job_class.Params.model_json_schema()
    for name, line in text["params"].items():
        schema["properties"][name]["description"] = line
    inputs = {name: {"kind": item.kind, "min": item.minimum, "max": item.maximum, "description": text["inputs"][name]}
              for name, item in job_class.inputs.items()}
    return {"description": text["description"], "params": schema, "inputs": inputs, "outputs": text["outputs"]}
