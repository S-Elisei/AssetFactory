"""Notifier: delivers the `notify_url` webhook of a job and of a batch. A delivery is an asyncio task that posts the JSON
`{"from": "asset-factory", "text": <text>}`. Every method runs on the one asyncio loop."""
import asyncio
import sys
from collections import Counter

import aiohttp

from core.store import FINISHED

# Seconds to wait after the first, second, ... failed attempt of a delivery; the last wait repeats. Guessed.
RETRY_DELAYS = (5, 15, 60, 300)
# Seconds one POST may take. Guessed.
POST_TIMEOUT = 10


def _reference(job):
    label = "" if job["label"] is None else f' "{job["label"]}"'
    return f"{job['job_id']}{label}"


def _work(job):
    text = f"work took {job['work_seconds']:.1f} s"
    if job["cloud_dollars"] > 0:
        text += f" ({job['cloud_dollars']:.3f} $)"
    return text


class Notifier:
    def __init__(self, store):
        self._store = store
        self._deliveries = set()

    def job_ended(self, job_id):
        """Sends the notification of the job to its `notify_url`, if it has one; and, when the job belongs to a batch
        with a `notify_url` whose jobs have all ended, the notification of the batch. The Runner calls it right after
        `Store.finish_job`, with no await between the two."""
        job = self._store.job(job_id)
        self._notify_job(job)
        if job["batch_id"] is not None:
            self._notify_batch(job["batch_id"])

    def jobs_stopped(self, job_ids):
        """Sends the notification of each job in `job_ids` that has a `notify_url`, and the notification of each batch of
        these jobs that has a `notify_url`, once per batch. The entry point calls it with the result of
        `Store.fail_unfinished`, after it has created the Notifier."""
        jobs = [self._store.job(job_id) for job_id in job_ids]
        for job in jobs:
            self._notify_job(job)
        for batch_id in dict.fromkeys(job["batch_id"] for job in jobs if job["batch_id"] is not None):
            self._notify_batch(batch_id)

    async def stop(self):
        """Cancels the pending deliveries."""
        for delivery in self._deliveries:
            delivery.cancel()
        await asyncio.gather(*self._deliveries, return_exceptions=True)

    def _notify_job(self, job):
        if job["notify_url"] is not None:
            lines = self._describe(job)
            self._send(job["notify_url"], "\n".join([f"AssetFactory job {_reference(job)}: {lines[0]}", *lines[1:]]))

    def _notify_batch(self, batch_id):
        batch = self._store.batch(batch_id)
        if batch["notify_url"] is not None and all(job["status"] in FINISHED for job in batch["jobs"]):
            self._send(batch["notify_url"], self._batch_text(batch))

    def _describe(self, job):
        """The summary line of the job, then its error line and its output lines."""
        lines = [f"{job['job']} {job['status']}, {_work(job)}."]
        if job["error"] is not None:
            lines.append(f"Error {job['error']['kind']}: {job['error']['message']}")
        files = self._store.output_files(job["job_id"])
        if files:
            lines.append("Output files:")
            lines += [f"  item {file['item']}: {file['path']} (file_id {file['file_id']})" for file in files]
        return lines

    def _batch_text(self, batch):
        counts = Counter(job["status"] for job in batch["jobs"])
        label = "" if batch["label"] is None else f' "{batch["label"]}"'
        summary = ", ".join(f"{counts[status]} {status}" for status in FINISHED if counts[status])
        lines = [f"AssetFactory batch {batch['batch_id']}{label} finished: {summary}."]
        for job in batch["jobs"]:
            job_lines = self._describe(job)
            lines += [f"- {_reference(job)}: {job_lines[0]}", *[f"  {line}" for line in job_lines[1:]]]
        return "\n".join(lines)

    def _send(self, url, text):
        delivery = asyncio.create_task(self._deliver(url, text))
        self._deliveries.add(delivery)
        delivery.add_done_callback(self._deliveries.discard)

    @staticmethod
    async def _deliver(url, text):
        """Posts `text` to `url`. A status below 500 ends the delivery (a 4xx is reported on stderr); a connection error,
        a timeout or a status of 500 or more is retried after the next of RETRY_DELAYS."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=POST_TIMEOUT)) as session:
            attempt = 0
            while True:
                try:
                    async with session.post(url, json={"from": "asset-factory", "text": text}) as response:
                        if response.status < 500:
                            if response.status >= 400:
                                print(f"notification to {url} rejected with HTTP {response.status}", file=sys.stderr)
                            return
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
                attempt += 1
