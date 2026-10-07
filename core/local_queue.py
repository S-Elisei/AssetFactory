"""LocalQueue: the queue of the local stage calls of the machine. It runs one call at a time and keeps at most one Worker
process alive; a call for a stage of another environment stops the alive Worker and starts the Worker of that
environment. Calls are served by priority (higher first), then by arrival; while the oldest waiting call of another
environment has waited less than AFFINITY_SECONDS, the calls for the alive Worker's environment go first. All methods
run on one asyncio loop, which must be able to run subprocesses (the Proactor loop on Windows)."""
import ast
import asyncio
import itertools
import json
import subprocess
import time
from contextlib import suppress
from functools import cache

import psutil

from core.calls import STOPPING, OutOfMemory, StageCancelled, StageFailed
from core.layout import DATA, ENVS, ROOT
from context import InputError

# Seconds an alive Worker stays up without a call. Guessed.
IDLE_SECONDS = 600.0
# Seconds the oldest waiting call of another environment may wait while the calls for the alive Worker's environment go
# first. Guessed.
AFFINITY_SECONDS = 180.0
# Seconds a Worker has to exit after its stdin is closed before it is killed. Guessed.
STOP_GRACE = 15.0
# Seconds a Worker has to end a cancelled run before it is killed. Guessed.
CANCEL_GRACE = 30.0
# Seconds between two checks of the free RAM for a call that waits for RAM.
RAM_CHECK_SECONDS = 1.0
# Bytes of the longest line the queue reads from a Worker. Guessed.
LINE_LIMIT = 2**24
LOGS = DATA / "logs"
# Bytes in one GB of RAM_GB.
GB = 1e9


@cache
def stage_path(stage):
    """The module `<root>/stages/<env>/<stage>.py` of the stage `stage`."""
    return next((ROOT / "stages").glob(f"*/{stage}.py"))


def stage_env(stage):
    """The environment of the stage `stage`."""
    return stage_path(stage).parent.name


@cache
def ram_need(stage):
    """The literal RAM_GB of the stage module, read from its source without importing it; None for a stage without
    RAM_GB."""
    tree = ast.parse(stage_path(stage).read_text(encoding="utf-8"))
    return next((ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "RAM_GB" for target in node.targets)), None)


class _Call:
    """One call: waiting, then starting (its Worker is being started), then running (its `run` message is sent). `outcome`
    is the result dict or the exception to raise once `ended` is set; `work` is the seconds of the run that the Worker
    reports. `seq` is also the id of its `run` message."""

    def __init__(self, stage, args, directory, priority, progress, charge, tag, seq):
        self.stage, self.env = stage, stage_env(stage)
        self.args, self.directory, self.priority, self.seq = args, directory, priority, seq
        self.progress, self.charge, self.tag = progress, charge, tag
        self.arrived = time.monotonic()
        self.phase = "waiting"
        self.cancelled = False
        self.ended = asyncio.Event()
        self.outcome = None
        self.worker = None
        self.started = None
        self.work = None
        self.fraction, self.message = 0.0, None
        self.ram = None

    def report(self, fraction, message):
        """Passes a progress message to the callback of the call."""
        self.fraction, self.message = fraction, message
        self.progress(fraction, message)


class _Worker:
    def __init__(self, env, process, log):
        self.env, self.process, self.log = env, process, log
        self.call = None
        self.reader = None


def _tree(worker):
    """The processes of the process tree of `worker` that still exist."""
    with suppress(psutil.NoSuchProcess):
        root = psutil.Process(worker.process.pid)
        return [root, *root.children(recursive=True)]
    return []


def _send(worker, message):
    worker.process.stdin.write((json.dumps(message) + "\n").encode("ascii"))
    return worker.process.stdin.drain()


def _end(call, outcome, seconds=None):
    """Ends `call` with `outcome` once; `seconds` is the time of its run that the Worker reports, if it does."""
    if not call.ended.is_set():
        call.outcome, call.work = outcome, seconds
        call.ended.set()


class LocalQueue:
    def __init__(self):
        self._waiting = []
        self._current = None
        self._worker = None
        self._task = None
        self._seq = itertools.count()
        self._wake = asyncio.Event()
        self._idle_since = time.monotonic()

    async def call(self, stage, args, directory, priority, progress, charge, tag):
        """Runs the local stage `stage` with `args` (JSON-able keyword arguments of its `run`) and returns the result dict
        of its `run`. `directory` is the existing folder for the stage's output files. A higher `priority` goes first.
        `tag` is opaque to the queue and is shown in `status()`.
        `progress(fraction, message)` is a synchronous callback that must not raise; it is called on the loop for each
        progress message of the run, also while a cancelled run ends, and, while the queue serves no other call, every
        RAM_CHECK_SECONDS while the call is short of RAM.
        `charge(seconds, dollars)` is a synchronous callback that must not raise; it is called once when the call
        returns or raises, whatever its outcome, with the seconds of the run (those the Worker reports for a result,
        else the time since the run was sent, 0 for a run never sent) and dollars 0.
        Raises InputError (the message of the stage), OutOfMemory, StageFailed (also when the Worker exits during the
        call) or StageCancelled (also when the queue is stopped). Cancelling the awaiting task cancels the call and
        re-raises the cancellation; a Worker that has not ended the cancelled run within CANCEL_GRACE is killed."""
        call = _Call(stage, args, directory, priority, progress, charge, tag, next(self._seq))
        self._waiting.append(call)
        if self._task is None:
            self._task = asyncio.create_task(self._serve())
        self._wake.set()
        try:
            try:
                await call.ended.wait()
            except asyncio.CancelledError:
                await self._cancel(call)
                raise
            if isinstance(call.outcome, Exception):
                raise call.outcome
            return call.outcome
        finally:
            self._settle(call)

    async def unload(self):
        """Stops the alive Worker. Returns False, and stops nothing, while a call is starting or running."""
        if self._current is not None:
            return False
        if self._worker is not None:
            await self._stop_worker()
        return True

    async def stop(self):
        """Ends the serving of calls: every waiting and starting call ends as StageCancelled, a running call is cancelled,
        and the alive Worker is stopped."""
        unserved = [*self._waiting, *([] if self._current is None else [self._current])]
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        for call in unserved:
            if call.phase != "running":
                _end(call, StageCancelled(STOPPING))
        self._waiting.clear()
        worker = self._worker
        if worker is not None:
            if worker.call is not None:
                await _send(worker, {"type": "cancel", "id": worker.call.seq})
            await self._stop_worker()

    def status(self):
        """Returns {"worker": {"env", "state": "idle" or "running"} or None, "running": {"stage", "tag", "priority",
        "phase": "starting" or "running", "fraction", "message"} or None, "waiting": [{"stage", "tag", "priority",
        "waited"}] in the order the calls are served (waited in seconds), "ram_waits": [{"stage", "tag", "need_gb",
        "available_gb"}] of the waiting calls that wait for RAM}."""
        now = time.monotonic()
        waiting = self._order()
        current = self._current
        worker = self._worker
        return {
            "worker": None if worker is None else {"env": worker.env,
                                                   "state": "idle" if worker.call is None else "running"},
            "running": None if current is None else {"stage": current.stage, "tag": current.tag,
                                                     "priority": current.priority, "phase": current.phase,
                                                     "fraction": current.fraction, "message": current.message},
            "waiting": [{"stage": call.stage, "tag": call.tag, "priority": call.priority,
                         "waited": round(now - call.arrived, 1)} for call in waiting],
            "ram_waits": [{"stage": call.stage, "tag": call.tag, "need_gb": call.ram[0],
                           "available_gb": round(call.ram[1], 1)} for call in waiting if call.ram is not None],
        }

    def _available(self):
        """Free system RAM in GB, plus the RSS of the alive Worker's process tree, which stopping the Worker frees."""
        available = psutil.virtual_memory().available
        if self._worker is not None:
            for process in _tree(self._worker):
                with suppress(psutil.NoSuchProcess):
                    available += process.memory_info().rss
        return available / GB

    def _order(self):
        """The waiting calls in the order they are served: by priority, then arrival; the calls for the alive Worker's
        environment first while the oldest call of another environment has waited less than AFFINITY_SECONDS."""
        ordered = sorted(self._waiting, key=lambda call: (-call.priority, call.seq))
        alive = None if self._worker is None else self._worker.env
        others = [call.arrived for call in ordered if call.env != alive]
        if alive is not None and others and time.monotonic() - min(others) < AFFINITY_SECONDS:
            ordered.sort(key=lambda call: call.env != alive)
        return ordered

    def _pick(self):
        """Returns the call to serve next, or None when every waiting call waits for RAM or none waits. A call that
        needs a new Worker for a stage with RAM_GB is picked only while the available RAM covers RAM_GB; a call that
        is short gets the RAM message and does not hold back the calls behind it."""
        ordered = self._order()
        for call in ordered:
            call.ram = None
        alive = None if self._worker is None else self._worker.env
        for call in ordered:
            need = None if call.env == alive else ram_need(call.stage)
            if need is None:
                return call
            available = self._available()
            if available >= need:
                return call
            call.ram = (need, available)
            call.report(0.0, f"waiting for system RAM: {call.stage} needs {need:.1f} GB, {available:.1f} GB "
                             "available")
        return None

    async def _serve(self):
        while True:
            self._wake.clear()
            call = self._pick()
            if call is not None:
                await self._execute(call)
                continue
            timeouts = []
            if any(call.ram is not None for call in self._waiting):
                timeouts.append(RAM_CHECK_SECONDS)
            if self._worker is not None:
                timeouts.append(self._idle_since + IDLE_SECONDS - time.monotonic())
            try:
                await asyncio.wait_for(self._wake.wait(), min(timeouts, default=None))
            except TimeoutError:
                if self._worker is not None and time.monotonic() - self._idle_since >= IDLE_SECONDS:
                    await self._stop_worker()

    async def _execute(self, call):
        """Gives `call` its Worker, sends its `run` and waits for its end."""
        self._waiting.remove(call)
        self._current = call
        call.phase = "starting"
        try:
            if self._worker is not None and self._worker.env != call.env:
                await self._stop_worker()
            if self._worker is None:
                self._worker = await self._start_worker(call.env)
            if call.cancelled:
                _end(call, StageCancelled("cancelled"))
                return
            worker = call.worker = self._worker
            worker.call = call
            call.phase = "running"
            call.started = time.monotonic()
            await _send(worker, {"type": "run", "id": call.seq, "stage": call.stage, "dir": str(call.directory),
                                 "args": call.args})
            await call.ended.wait()
        finally:
            self._current = None
            self._idle_since = time.monotonic()

    async def _cancel(self, call):
        """Cancels `call`: a waiting call is removed; a running call gets a `cancel` message, and its Worker is killed
        when the run has not ended within CANCEL_GRACE."""
        if call.phase == "waiting":
            self._waiting.remove(call)
            return
        call.cancelled = True
        if call.phase == "running" and not call.ended.is_set():
            await _send(call.worker, {"type": "cancel", "id": call.seq})
            try:
                await asyncio.wait_for(call.ended.wait(), CANCEL_GRACE)
            except TimeoutError:
                await self._kill(call.worker)
                await call.worker.reader

    @staticmethod
    def _settle(call):
        """Calls the `charge` of `call` with the seconds of its run: those the Worker reports, else the time since the run
        was sent, 0 for a run never sent."""
        if call.work is None:
            call.work = 0.0 if call.started is None else time.monotonic() - call.started
        call.charge(call.work, 0.0)

    async def _start_worker(self, env):
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / f"worker-{env}.log"
        with open(log, "ab") as stderr:
            process = await asyncio.create_subprocess_exec(
                str(ENVS / env / "Scripts" / "python.exe"), str(ROOT / "worker" / "main.py"), env, cwd=ROOT,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr, limit=LINE_LIMIT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        worker = _Worker(env, process, log)
        worker.reader = asyncio.create_task(self._read(worker))
        return worker

    async def _read(self, worker):
        """Follows the messages of `worker` until its stdout closes, then fails the call it was running."""
        while line := await worker.process.stdout.readline():
            message = json.loads(line)
            call = worker.call
            if message["type"] == "progress":
                call.report(message["fraction"], message["message"])
                continue
            worker.call = None
            if message["type"] == "done":
                _end(call, message["result"], message["seconds"])
            else:
                _end(call, self._error(worker, message))
        code = await worker.process.wait()
        if self._worker is worker:
            self._worker = None
        if worker.call is not None:
            _end(worker.call, StageFailed(f"the {worker.env} Worker exited with code {code} while running "
                                          f"{worker.call.stage}; its log is {worker.log}"))
            worker.call = None

    @staticmethod
    def _error(worker, message):
        kind, text = message["kind"], message["message"]
        if kind == "input":
            return InputError(text)
        if kind == "oom":
            return OutOfMemory(f"{text}; the traceback is in {worker.log}")
        if kind == "cancelled":
            return StageCancelled(text)
        return StageFailed(f"{text}; the traceback is in {worker.log}")

    async def _stop_worker(self):
        """Closes the stdin of the alive Worker and waits STOP_GRACE seconds for it to exit, then kills it."""
        worker, self._worker = self._worker, None
        worker.process.stdin.close()
        try:
            await asyncio.wait_for(worker.process.wait(), STOP_GRACE)
        except TimeoutError:
            await self._kill(worker)
        await worker.reader

    @staticmethod
    async def _kill(worker):
        """Kills the process tree of `worker`."""
        for process in _tree(worker):
            with suppress(psutil.NoSuchProcess):
                process.kill()
        await worker.process.wait()
