"""Api: the FastAPI application of the factory. Every handler is `async def` and runs on the one asyncio loop, which owns
the connection of the Store. The endpoints are described to agents in `usage.md`, which `GET /api/usage` serves with the
description of every job. The UI is served from `<root>/ui` at `/`.

A request that is refused answers `{"error": {"message": <state and next action>, "details": [<defect>, ...]}}`. A
request that does not match the request schema lists the defects of the schema. A request that does is checked
completely - the params of the job, its files, and the installation of its stages - and every defect is listed; the
job is created in the same step that follows the checks, with no await between."""
import asyncio
import json
import random
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import psutil
import pynvml
from fastapi import FastAPI, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException

from core.cloud_queue import SPEND_REFRESH_SECONDS
from core.job import SEED_MAX, describe, job_names, load_job
from core.layout import ROOT
from core.local_queue import GB, stage_env
from core.readiness import app_readiness, stage_readiness
from core.store import FINISHED, Refused

UI = ROOT / "ui"
# Limits of the query parameters: the longest wait in seconds, the largest and the default page of the job history.
WAIT_MAX = 120
PAGE_MAX = 200
PAGE_SIZE = 50
# Seconds between two keepalive lines of the event stream. Guessed.
KEEPALIVE_SECONDS = 15

INVALID = "the request is refused; correct every defect listed in details, then send it again; the fields are " \
          "described in GET /api/usage"
NOT_INSTALLED = "the request is refused; run the command in details from the repository root, then send the " \
                "request again"

Status = Literal["queued", "running", *FINISHED]
Wait = Annotated[float, Query(ge=0, le=WAIT_MAX)]


def _http_url(value):
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("must be an http or https URL, for example http://127.0.0.1:9000/hook")
    return value


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job: str
    params: dict
    count: int = Field(ge=1)
    notify_url: Annotated[str, AfterValidator(_http_url)] | None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobs: list[JobRequest] = Field(min_length=1)
    notify_url: Annotated[str, AfterValidator(_http_url)] | None


class Refusal(Exception):
    """A refused request: `status` is the HTTP status, `message` names the state and the next action, `details` lists
    the defects."""

    def __init__(self, status, message, details=()):
        super().__init__(message)
        self.status, self.message, self.details = status, message, list(details)


def _error(status, message, details=()):
    return JSONResponse({"error": {"message": message, "details": list(details)}}, status_code=status)


async def _refusal(request, error):
    return _error(error.status, error.message, error.details)


async def _invalid(request, error):
    """The defects of a request that does not match the request schema; the first part of a location is `body`,
    `query` or `path`."""
    paths = [(_path(*defect["loc"][1:]), defect["msg"]) for defect in error.errors()]
    return _error(400, INVALID, [f"{path}: {msg}" if path else msg for path, msg in paths])


async def _http_error(request, error):
    if error.status_code in (404, 405):
        return _error(error.status_code, f"{request.method} {request.url.path} is not an endpoint of the factory; "
                                         "the endpoints are described in GET /api/usage")
    return _error(error.status_code, error.detail)


def _path(*parts):
    """The path of a field, as `jobs[2].params.steps` from ("jobs", 2, "params", "steps"); empty parts are skipped."""
    text = ""
    for part in parts:
        if isinstance(part, int):
            text += f"[{part}]"
        elif part:
            text += f".{part}" if text else part
    return text


def _job_section(name, info):
    """The markdown section of a job from its `describe()` data: description, modality, params schema, outputs."""
    outputs = [f"- `{file}`: {text}" for file, text in info["outputs"].items()]
    return "\n".join([f"### {name}", "", info["description"], "", f"Modality: {info['modality']}", "", "Params (JSON schema):",
                      "", "```json",
                      json.dumps(info["params"], separators=(",", ":")), "```", "", "Outputs:", "", *outputs, ""])


class Api:
    def __init__(self, store, runner, local_queue, cloud_queue, lifespan):
        """Builds the application, `self.app`. `lifespan` is the lifespan context manager of the application."""
        self._store, self._runner = store, runner
        self._local, self._cloud = local_queue, cloud_queue
        self._jobs = {name: load_job(name) for name in job_names()}
        self._schema = {name: describe(job_class) for name, job_class in self._jobs.items()}
        # The extensions of all file params of all jobs: the extensions an upload may have.
        self._uploads = sorted({extension for job_class in self._jobs.values()
                                for extensions in job_class.file_params().values() for extension in extensions})
        pynvml.nvmlInit()
        self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        guide = Path(__file__).with_name("usage.md").read_text(encoding="utf-8")
        limits = {"UPLOADS": ", ".join(self._uploads), "SEED_MAX": SEED_MAX, "WAIT_MAX": WAIT_MAX, "PAGE_MAX": PAGE_MAX,
                  "PAGE_SIZE": PAGE_SIZE, "SPEND_REFRESH_SECONDS": SPEND_REFRESH_SECONDS}
        for key, value in limits.items():
            guide = guide.replace("{{" + key + "}}", str(value))
        self._guide = guide + "\n".join(_job_section(name, info) for name, info in self._schema.items())

        app = self.app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
        app.add_exception_handler(Refusal, _refusal)
        app.add_exception_handler(RequestValidationError, _invalid)
        app.add_exception_handler(HTTPException, _http_error)
        for method, path, handler, status in (
                ("GET", "/api/usage", self.usage, 200),
                ("GET", "/api/health", self.health, 200),
                ("GET", "/api/jobs/schema", self.jobs_schema, 200),
                ("POST", "/api/files", self.upload_file, 201),
                ("GET", "/api/files", self.list_files, 200),
                ("GET", "/api/files/{file_id}", self.get_file, 200),
                ("GET", "/api/files/{file_id}/content", self.file_content, 200),
                ("POST", "/api/jobs", self.create_job, 201),
                ("GET", "/api/jobs", self.list_jobs, 200),
                ("GET", "/api/jobs/{job_id}", self.get_job, 200),
                ("POST", "/api/jobs/{job_id}/cancel", self.cancel_job, 202),
                ("POST", "/api/jobs/{job_id}/retry", self.retry_job, 201),
                ("DELETE", "/api/jobs/{job_id}", self.delete_job, 200),
                ("POST", "/api/batches", self.create_batch, 201),
                ("GET", "/api/batches/{batch_id}", self.get_batch, 200),
                ("GET", "/api/system", self.system, 200),
                ("POST", "/api/system/unload", self.unload, 200),
                ("GET", "/api/events", self.events, 200)):
            app.add_api_route(path, handler, methods=[method], status_code=status)
        app.mount("/", StaticFiles(directory=UI, html=True))

    # Guide and health.

    async def usage(self, request: Request):
        base = str(request.base_url).rstrip("/")
        return PlainTextResponse(self._guide.replace("{{BASE}}", base), media_type="text/markdown; charset=utf-8")

    async def health(self):
        return {"status": "ok"}

    async def jobs_schema(self):
        return self._schema

    # Files.

    async def upload_file(self, file: UploadFile):
        if Path(file.filename).suffix.lower() not in self._uploads:
            raise Refusal(400, f"{file.filename}: this extension is not accepted; upload a file with one of these "
                               f"extensions: {', '.join(self._uploads)}")
        return self._store.add_upload(file.filename, await file.read())

    async def list_files(self, origin: Literal["upload", "output"] | None = None):
        return {"files": self._store.files(origin)}

    async def get_file(self, file_id: str):
        return self._found(self._store.file(file_id), "file", file_id, "GET /api/files")

    async def file_content(self, file_id: str):
        file = self._found(self._store.file(file_id), "file", file_id, "GET /api/files")
        return FileResponse(file["path"], filename=file["name"], content_disposition_type="inline")

    # Jobs.

    async def create_job(self, request: JobRequest):
        fields, invalid, missing = self._prepare(request, "")
        self._refuse(invalid, missing)
        job = self._store.create_job(**fields)
        self._runner.submit(job["job_id"])
        return self._view(job)

    async def list_jobs(self, status: Status | None = None, job: str | None = None, batch_id: str | None = None,
                        before: int | None = None, limit: Annotated[int, Query(ge=1, le=PAGE_MAX)] = PAGE_SIZE):
        jobs = self._store.jobs(status, None if job is None else job.split(","), batch_id, before, limit)
        return {"jobs": self._views(jobs), "next_before": jobs[-1]["seq"] if len(jobs) == limit else None}

    async def get_job(self, job_id: str, wait: Wait = 0):
        self._found(self._store.job(job_id), "job", job_id, "GET /api/jobs")
        await self._wait(lambda: self._store.job(job_id)["status"] in FINISHED, wait)
        return self._view(self._store.job(job_id))

    async def cancel_job(self, job_id: str):
        job = self._found(self._store.job(job_id), "job", job_id, "GET /api/jobs")
        if job["status"] in FINISHED:
            raise Refusal(409, f"job {job_id} has already ended as {job['status']}; only a queued or running job "
                               "can be cancelled")
        self._runner.cancel(job_id)
        return self._view(job)

    async def retry_job(self, job_id: str):
        job = self._found(self._store.job(job_id), "job", job_id, "GET /api/jobs")
        params = job["params"]
        if job["status"] not in FINISHED:
            raise Refusal(409, f"job {job_id} is {job['status']}; only a job that has ended can be retried; wait for "
                               "it to end or cancel it, then retry")
        if job["status"] == "succeeded":
            if "seed" not in params:
                raise Refusal(409, f"job {job_id} succeeded and has no seed param; create a new job with "
                                   "POST /api/jobs")
            params = {**params, "seed": "random"}
        for name in self._jobs[job["job"]].file_params():
            for file_id in params[name]:
                if self._store.file(file_id) is None:
                    raise Refusal(409, f"param {name} of job {job_id} names file {file_id}, which no longer exists; "
                                       "create a new job with existing files")
        fields, invalid, missing = self._prepare(JobRequest(
            job=job["job"], params=params, count=job["count"], notify_url=job["notify_url"]), "")
        self._refuse(invalid, missing)
        new = self._store.create_job(**fields)
        self._runner.submit(new["job_id"])
        return self._view(new)

    async def delete_job(self, job_id: str):
        self._found(self._store.job(job_id), "job", job_id, "GET /api/jobs")
        try:
            self._store.delete_job(job_id)
        except Refused as error:
            raise Refusal(409, str(error))
        self._runner.publish(job_id)
        return {"deleted": job_id}

    # Batches.

    async def create_batch(self, request: BatchRequest):
        results = [self._prepare(job, f"jobs[{index}]") for index, job in enumerate(request.jobs)]
        self._refuse([text for _, invalid, _ in results for text in invalid],
                     [text for _, _, missing in results for text in missing])
        batch = self._store.create_batch(request.notify_url, [fields for fields, _, _ in results])
        for job in batch["jobs"]:
            self._runner.submit(job["job_id"])
        return {**batch, "jobs": self._views(batch["jobs"])}

    async def get_batch(self, batch_id: str, wait: Wait = 0):
        self._found(self._store.batch(batch_id), "batch", batch_id, "POST /api/batches")
        await self._wait(lambda: all(job["status"] in FINISHED for job in self._store.batch(batch_id)["jobs"]), wait)
        batch = self._store.batch(batch_id)
        return {**batch, "jobs": self._views(batch["jobs"])}

    # System.

    async def system(self):
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._gpu)
        ram = psutil.virtual_memory()
        return {"gpu": {"name": pynvml.nvmlDeviceGetName(self._gpu), "vram_used_gb": round(memory.used / GB, 2),
                        "vram_total_gb": round(memory.total / GB, 2),
                        "utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu},
                "cpu": {"percent": psutil.cpu_percent()}, "cloud": self._cloud.spend,
                "ram": {"total_gb": round(ram.total / GB, 2), "available_gb": round(ram.available / GB, 2)},
                "local_queue": self._local.status(), "cloud_queue": self._cloud.status()}

    async def unload(self):
        if not await self._local.unload():
            raise Refusal(409, "a stage is starting or running in the local queue; wait for it to end or cancel "
                               "its job, then unload again")
        return self._local.status()

    async def events(self):
        queue = self._runner.subscribe()

        async def stream():
            try:
                while True:
                    try:
                        job_id = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    job = self._store.job(job_id)
                    if job is None:
                        yield f"event: deleted\ndata: {json.dumps(job_id)}\n\n"
                    else:
                        yield f"data: {json.dumps(self._view(job))}\n\n"
            finally:
                self._runner.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # Internals.

    @staticmethod
    def _found(row, kind, row_id, source):
        if row is None:
            raise Refusal(404, f"no {kind} {row_id}; ids come from {source}")
        return row

    @staticmethod
    def _refuse(invalid, missing):
        """Raises the refusal for the defects of a request: 400 when any defect is not an installation, else 409."""
        if invalid:
            raise Refusal(400, INVALID, [*invalid, *missing])
        if missing:
            raise Refusal(409, NOT_INSTALLED, missing)

    def _views(self, jobs):
        """The job objects: the rows of the Store with the progress of the Runner."""
        progress = self._runner.progress()
        return [{**job, "progress": progress.get(job["job_id"])} for job in jobs]

    def _view(self, job):
        return self._views([job])[0]

    async def _wait(self, ended, seconds):
        """Returns when `ended()` is true or after `seconds`, whichever comes first."""
        queue = self._runner.subscribe()
        try:
            async with asyncio.timeout(seconds):
                while not ended():
                    await queue.get()
        except TimeoutError:
            pass
        finally:
            self._runner.unsubscribe(queue)

    def _prepare(self, request, where):
        """Checks the job request `request`, whose path in the body is `where`. Returns (fields, invalid, missing):
        the keyword arguments of `Store.create_job` (usable when both lists are empty), the texts of the defects of the
        request, and the texts of the installs that the job's stages and Modal apps need."""
        job_class = self._jobs.get(request.job)
        if job_class is None:
            return None, [f"{_path(where, 'job')}: no job named {request.job}; the jobs are listed in "
                          "GET /api/usage"], []
        invalid = []
        params = None
        try:
            params = job_class.Params.model_validate(request.params).model_dump()
        except ValidationError as error:
            invalid += [f"{_path(where, 'params', *defect['loc'])}: {defect['msg']}" for defect in error.errors()]
        else:
            invalid += self._file_defects(job_class, params, where)
        commands = {}
        for kind, name in job_class.stages():
            if kind == "local":
                command, label = stage_readiness(stage_env(name), name), f"stage {name}"
            else:
                command, label = app_readiness(name), f"Modal app {name}"
            if command is not None:
                commands.setdefault(command, []).append(label)
        missing = [f"{_path(where, 'job')}: {request.job} cannot run; not installed: {', '.join(labels)}. From the "
                   f"repository root run: {command}" for command, labels in commands.items()]
        if params is not None and params.get("seed") == "random":
            params["seed"] = random.randint(0, SEED_MAX)
        fields = {"job": request.job, "params": params, "count": request.count, "notify_url": request.notify_url}
        return fields, invalid, missing

    def _file_defects(self, job_class, params, where):
        """The texts of the defects of the file params in the validated `params`: a file that does not exist, or whose
        extension is not among the accepted extensions of the param."""
        defects = []
        for name, extensions in job_class.file_params().items():
            for index, file_id in enumerate(params[name]):
                where_file = _path(where, "params", name, index)
                file = self._store.file(file_id)
                if file is None:
                    defects.append(f"{where_file}: no file {file_id}; use a file_id from POST /api/files or "
                                   "GET /api/files")
                elif Path(file["name"]).suffix.lower() not in extensions:
                    defects.append(f"{where_file}: file {file_id} is named {file['name']}, the param takes files with "
                                   f"these extensions: {', '.join(extensions)}")
        return defects
