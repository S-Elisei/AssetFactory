"""CloudQueue: one lane per Modal app of `cloud/`, all running in parallel. A lane has at most one call computing in
Modal. It sends the next waiting call (higher priority first, then arrival) when the computing call is ready: its
`cloud.DONE` message has arrived in its progress queue, else its result has been received, or it has failed or been
cancelled. Once the `DONE` message has arrived, the download of the result holds the lane no longer. All methods run on
one asyncio loop.

Cost. A call is charged (t1 - t0) seconds at the rate of the app, `GPU_RATES[GPU] + CPU_RATE * CPU + MEMORY_RATE *
MEMORY_MIB / 1024` of its module. t0 is the time the call was sent; t1 is the time of its `DONE` message, else of its
result, else of its failure or cancellation. A call that was never sent is charged nothing. The work seconds of a call
are the `seconds` of its result, else t1 - t0."""
import asyncio
import importlib
import itertools
import time
from pathlib import Path

import cloud
import modal
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal.config import config
from modal_proto import api_pb2

from core.calls import STOPPING, OutOfMemory, StageCancelled, StageFailed
from core.install import cloud_apps
from core.readiness import app_readiness
from context import InputError

# Dollars per second of one GPU of each type. Documented: https://modal.com/pricing.
GPU_RATES = {"L40S": 0.000542}
# Dollars per second of one physical CPU core, and of one GiB of memory. Documented: https://modal.com/pricing.
CPU_RATE = 0.0000131
MEMORY_RATE = 0.00000222
# Seconds of one wait for the result of a call, between two reads of its progress queue.
POLL_SECONDS = 1


@synchronizer.wrap
async def _stop_containers(app_id):
    """Stops every container of the Modal app `app_id`: lists them with TaskList and stops each with ContainerStop."""
    client = await _Client.from_env()
    listing = await client._stub.TaskList(
        api_pb2.TaskListRequest(environment_name=config.get("environment") or "", app_id=app_id))
    for task in listing.tasks:
        await client._stub.ContainerStop(api_pb2.ContainerStopRequest(task_id=task.task_id))


def _read(files):
    return {name: Path(path).read_bytes() for name, path in files.items()}


def _write(value, directory):
    """Writes every `bytes` value of the dict `value` to the file of its key in `directory`; returns the dict with those
    values replaced by the paths."""
    result = {}
    for key, item in value.items():
        if isinstance(item, bytes):
            path = Path(directory) / key
            path.write_bytes(item)
            item = str(path)
        result[key] = item
    return result


def _order(call):
    """The key of the order in which the waiting calls of a lane are sent: priority (higher first), then arrival."""
    return -call.priority, call.seq


class _Call:
    """One call: waiting, then sent (a member of the lane's `sent` from its admission to its end). `t0` and `t1` are the
    times of the cost rule; `result` is the result of the Modal call."""

    def __init__(self, files, params, priority, progress, charge, tag, seq):
        self.files, self.params, self.priority, self.seq = files, params, priority, seq
        self.progress, self.charge, self.tag = progress, charge, tag
        self.arrived = time.time()
        self.admitted = asyncio.Event()
        self.function_call = None
        self.t0 = self.t1 = None
        self.result = None
        self.cancelled = False
        self.fraction, self.message = 0.0, None


class _Lane:
    def __init__(self, app):
        module = importlib.import_module(f"cloud.{app}")
        self.app = app
        self.name = module.APP_NAME
        self.run = modal.Cls.from_name(module.APP_NAME, module.CLASS)().run
        self.rate = GPU_RATES[module.GPU] + CPU_RATE * module.CPU + MEMORY_RATE * module.MEMORY_MIB / 1024
        self.waiting, self.sent = [], []
        self.computing = None
        self.seq = itertools.count()
        self.closed = False
        self.seconds = self.dollars = 0.0

    async def call(self, files, params, directory, priority, progress, charge, tag):
        call = _Call(files, params, priority, progress, charge, tag, next(self.seq))
        self.waiting.append(call)
        self._dispatch()
        try:
            try:
                call.result = await self._run(call)
            except asyncio.CancelledError:
                await self._cancel(call)
                raise
            except Exception as error:
                failure = self._failure(call, error)
                if not isinstance(error, InputError) and not call.cancelled:
                    await self._cancel(call)
                raise failure from (None if failure is error else error)
            return await asyncio.to_thread(_write, call.result, directory)
        finally:
            if call.t1 is None:
                call.t1 = time.time()
            for calls in (self.waiting, self.sent):
                if call in calls:
                    calls.remove(call)
            self._release(call)
            self._settle(call)

    def _dispatch(self):
        """Sends the best waiting call when the lane is open and no call is computing."""
        if self.closed or not self.waiting or self.computing is not None:
            return
        call = min(self.waiting, key=_order)
        self.waiting.remove(call)
        self.sent.append(call)
        self.computing = call
        call.admitted.set()

    def _release(self, call):
        """Frees the lane when `call` is the computing call, and sends the next call."""
        if self.computing is call:
            self.computing = None
            self._dispatch()

    async def _run(self, call):
        """Waits for the admission of `call`, sends it and returns the result of the Modal call."""
        await call.admitted.wait()
        if call.cancelled:
            raise StageCancelled(STOPPING)
        contents = await asyncio.to_thread(_read, call.files)
        async with modal.Queue.ephemeral() as queue:
            call.t0 = time.time()
            call.function_call = await self.run.spawn.aio(**contents, params=call.params, progress=queue)
            return await self._poll(call, queue)

    async def _poll(self, call, queue):
        """Forwards the progress of the call and returns its result."""
        while True:
            await self._forward(call, queue)
            try:
                value = await call.function_call.get.aio(timeout=POLL_SECONDS)
                break
            except TimeoutError:
                continue
        if call.t1 is None:
            call.t1 = time.time()
        self._release(call)
        await self._forward(call, queue)
        return value

    async def _forward(self, call, queue):
        """Passes the messages in the progress queue of the call to its callback. The `DONE` message sets `t1` and frees
        the lane."""
        for fraction, message in await queue.get_many.aio(100, block=False):
            if (fraction, message) == cloud.DONE:
                if call.t1 is None:
                    call.t1 = time.time()
                self._release(call)
                continue
            call.fraction, call.message = fraction, message
            call.progress(fraction, message)

    async def _cancel(self, call):
        """Cancels the Modal call of `call` with a plain cancel while `call` holds the lane."""
        if call.function_call is not None and self.computing is call:
            await call.function_call.cancel.aio()

    @staticmethod
    def _failure(call, error):
        """Returns the exception that ends `call` after `error`."""
        if call.cancelled:
            return StageCancelled(STOPPING)
        if isinstance(error, InputError):
            return error
        if "CUDA out of memory" in str(error):
            return OutOfMemory(str(error))
        return StageFailed(f"{type(error).__name__}: {error}")

    def _settle(self, call):
        """Charges `call` by the cost rule, adds the charge to the totals of the lane and calls the `charge` of the call."""
        seconds = dollars = 0.0
        if call.t0 is not None:
            span = call.t1 - call.t0
            seconds = span if call.result is None else call.result["seconds"]
            dollars = span * self.rate
        self.seconds += seconds
        self.dollars += dollars
        call.charge(seconds, dollars)

    def status(self):
        now = time.time()

        def view(call):
            return {"tag": call.tag, "priority": call.priority, "waited": round(now - call.arrived, 1),
                    "fraction": call.fraction, "message": call.message}

        return {"running": None if self.computing is None else view(self.computing),
                "waiting": [view(call) for call in sorted(self.waiting, key=_order)],
                "seconds": round(self.seconds, 1), "dollars": round(self.dollars, 4)}

    def close(self):
        """Stops sending calls and marks every call of the lane cancelled; the waiting calls are woken and end as
        cancelled. Returns the cancels, with the container terminated, of the sent calls that have a Modal call."""
        self.closed = True
        for call in (*self.waiting, *self.sent):
            call.cancelled = True
            call.admitted.set()
        return [call.function_call.cancel.aio(terminate_containers=True)
                for call in self.sent if call.function_call is not None]


class CloudQueue:
    def __init__(self):
        self._lanes = {app: _Lane(app) for app in cloud_apps()}

    async def call(self, app, files, params, directory, priority, progress, charge, tag):
        """Runs the call of the Modal app `app` (the module name in `cloud/`) and returns its result dict: every `bytes`
        value written to the file of its key in `directory` and replaced by its path. `files` is {argument name: path of
        the file}, read as bytes; `params` is the params dict. A higher `priority` is sent first. `tag` is opaque to the
        queue and is shown in `status()`.
        `progress(fraction, message)` is a synchronous callback that must not raise; it is called on the loop for each
        message of the container until the call is settled.
        `charge(seconds, dollars)` is a synchronous callback that must not raise; it is called once when the call is
        settled, whatever its outcome (see the cost rule of the module).
        Raises InputError (the message of the container), OutOfMemory, StageFailed (any other exception of the
        container, or of a Modal call, which is never retried) or StageCancelled (the queue was stopped).
        Cancel rule: when the awaiting task is cancelled, or the call ends with an exception other than InputError, and
        the call still holds the lane (its `DONE` message has not arrived), the Modal call gets a plain cancel, without
        `terminate_containers`; after `DONE` nothing is cancelled. A cancellation of the task is re-raised. A cancel
        that happens while the Modal call is being sent can leave that Modal call running. Only `stop()` terminates
        containers."""
        return await self._lanes[app].call(files, params, directory, priority, progress, charge, tag)

    def status(self):
        """Returns {app: {"running": a call or None, "waiting": [call], "seconds", "dollars"}}; a call is {"tag",
        "priority", "waited", "fraction", "message"}; `running` is the computing call; `seconds` and `dollars` are the
        totals of the app since the start."""
        return {app: lane.status() for app, lane in self._lanes.items()}

    async def stop(self):
        """Stops sending calls, ends the waiting calls as cancelled, cancels every sent call that has a Modal call with
        its container terminated, then stops the containers of the ready Modal apps."""
        lanes = list(self._lanes.values())
        await asyncio.gather(*(cancel for lane in lanes for cancel in lane.close()))
        await asyncio.gather(*(self._stop_app(lane.name) for lane in lanes if app_readiness(lane.app) is None))

    @staticmethod
    async def _stop_app(name):
        app = await modal.App.lookup.aio(name)
        await _stop_containers.aio(app.app_id)
