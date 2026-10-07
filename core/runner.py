"""Runner: runs every job as an asyncio task and keeps its status in the Store. A job of `count` items runs its `run`
once per item, all items at the same time; item `i` has the `seed` param of the job plus `i`. The first item that fails
ends the job and cancels the other items. All methods run on the one asyncio loop.

A job ends as
- succeeded, when every item returned;
- failed, with the error kind `input` (InputError, the message of the stage), `oom` (OutOfMemory) or `failed`
  (StageFailed);
- cancelled, when `cancel` was called.
A job is `queued` until the first progress message of its first call arrives, then `running`. A job that `stop`
cancelled keeps its status. The progress of a job that has not ended is {"fraction", "message"} and is kept in memory
only."""
import asyncio
from dataclasses import dataclass

from core.calls import OutOfMemory, StageFailed
from core.job import Context, load_job
from context import InputError


@dataclass
class _Progress:
    items: list
    message: str = None
    started: bool = False

    @property
    def fraction(self):
        return sum(self.items) / len(self.items)


class Runner:
    def __init__(self, store, notifier, local_queue, cloud_queue):
        self._store, self._notifier = store, notifier
        self._local, self._cloud = local_queue, cloud_queue
        self._tasks = {}
        self._progress = {}
        self._subscribers = set()
        self._stopping = False

    def submit(self, job_id):
        """Starts the job `job_id`, which is queued in the Store."""
        self._tasks[job_id] = asyncio.create_task(self._run(job_id))
        self.publish(job_id)

    def cancel(self, job_id):
        """Cancels the running job `job_id`; it ends as cancelled."""
        self._tasks[job_id].cancel()

    def progress(self):
        """Returns {job_id: {"fraction", "message"}} for every job that has not ended."""
        return {job_id: {"fraction": progress.fraction, "message": progress.message}
                for job_id, progress in self._progress.items()}

    def subscribe(self):
        """Returns an asyncio.Queue that receives the job_id of every job that changes: when it is submitted, starts,
        reports progress and ends, and of every job that is deleted. The subscriber reads the job from the Store, where a
        deleted job has no row, and its progress from `progress()`. The caller calls `unsubscribe` when it stops
        reading."""
        queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self._subscribers.discard(queue)

    async def stop(self):
        """Cancels every job task without changing the status of its row. The entry point stops the Runner first, then the
        queues, then the Notifier."""
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def publish(self, job_id):
        """Sends the job_id to every subscriber."""
        for queue in self._subscribers:
            queue.put_nowait(job_id)

    def _finish(self, job_id, status, kind=None, message=None):
        """Ends the job in the Store and notifies, with no await between."""
        self._store.finish_job(job_id, status, kind, message)
        self._notifier.job_ended(job_id)
        self.publish(job_id)

    async def _run(self, job_id):
        job = self._store.job(job_id)
        self._progress[job_id] = _Progress([0.0] * job["count"])
        try:
            await self._run_items(job, load_job(job["job"]))
        except InputError as error:
            self._finish(job_id, "failed", "input", str(error))
        except OutOfMemory as error:
            self._finish(job_id, "failed", "oom", str(error))
        except StageFailed as error:
            self._finish(job_id, "failed", "failed", str(error))
        except asyncio.CancelledError:
            if not self._stopping:
                self._finish(job_id, "cancelled")
        else:
            self._finish(job_id, "succeeded")
        finally:
            del self._tasks[job_id], self._progress[job_id]

    async def _run_items(self, job, job_class):
        params = job_class.Params(**job["params"])
        inputs = {name: [self._store.file(file_id)["path"] for file_id in job["inputs"].get(name, [])]
                  for name in job_class.inputs}
        stage_count = len(job_class.stages())
        progress = self._progress[job["job_id"]]

        def report(item, fraction, message):
            if not progress.started:
                progress.started = True
                self._store.start_job(job["job_id"])
            progress.items[item] = fraction
            progress.message = message if job["count"] == 1 else f"item {item}: {message}"
            self.publish(job["job_id"])

        async def run_item(item):
            context = Context(self._store, self._local, self._cloud, job, params, inputs, item, stage_count, report)
            await job_class().run(context)
            report(item, 1.0, "done")

        items = [asyncio.create_task(run_item(item)) for item in range(job["count"])]
        try:
            await asyncio.gather(*items)
        except BaseException:
            for item in items:
                item.cancel()
            await asyncio.gather(*items, return_exceptions=True)
            raise
