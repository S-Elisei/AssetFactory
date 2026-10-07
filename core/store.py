"""Store: the SQLite database and the files of the factory, under `DATA`. All methods are synchronous and run on the
one asyncio loop; there is one connection. The schema is created when the database file does not exist.

Files. `data/files/<file_id><suffix>` holds an upload; `data/jobs/<job_id>/` is the folder of a job and holds its
outputs, wherever below it they lie. A file row: file_id, kind, origin (`upload` when job_id is None, else `output`),
name, path (absolute), size, created, job_id and item (None for an upload). `kind` is opaque to the Store.

Jobs. A job row: seq (the paging cursor), job_id, job (the file name of its module in `jobs/`), params (dict), inputs
(dict of input name to the list of its file_ids), count, seed, priority, label, notify_url, batch_id, status, error (None,
or a dict with `kind` and `message`), outputs (per item of `count`, the list of the file_ids of that item), created,
started, finished (epoch seconds, None until they happen), work_seconds and cloud_dollars. Statuses: queued, running,
then one of FINISHED. Methods that take a job_id or a file_id require an id that exists; `job`, `file` and `batch`
return None for an unknown id."""
import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from core.layout import DATA

FINISHED = ("succeeded", "failed", "cancelled")

SCHEMA = """
CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    label TEXT,
    notify_url TEXT,
    created REAL NOT NULL
);
CREATE TABLE jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    job TEXT NOT NULL,
    params TEXT NOT NULL,
    inputs TEXT NOT NULL,
    count INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    priority INTEGER NOT NULL,
    label TEXT,
    notify_url TEXT,
    batch_id TEXT,
    status TEXT NOT NULL,
    error_kind TEXT,
    error_message TEXT,
    created REAL NOT NULL,
    started REAL,
    finished REAL,
    work_seconds REAL NOT NULL DEFAULT 0,
    cloud_dollars REAL NOT NULL DEFAULT 0
);
CREATE INDEX jobs_batch ON jobs (batch_id);
CREATE TABLE files (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    created REAL NOT NULL,
    job_id TEXT,
    item INTEGER
);
CREATE INDEX files_job ON files (job_id);
"""


class Refused(Exception):
    """The request names a state of the Store that does not allow it. The message names the state and the next action."""


def _new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self):
        """Opens the database `DATA/factory.db` and creates `DATA` and the schema when the file is absent."""
        database = DATA / "factory.db"
        new = not database.exists()
        DATA.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(database)
        self._db.row_factory = sqlite3.Row
        if new:
            self._db.executescript(SCHEMA)

    # Files.

    def add_upload(self, name, kind, content):
        """Writes `content` (bytes) as an upload named `name` and returns its file row."""
        name = Path(name).name
        file_id = _new_id("f")
        path = DATA / "files" / f"{file_id}{Path(name).suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return self._insert_file(file_id, kind, name, path, None, None)

    def add_output(self, job_id, item, path, kind):
        """Registers the existing file `path`, which lies below `job_dir(job_id)`, as an output of item `item` of the
        job, named by its file name. Returns its file row."""
        return self._insert_file(_new_id("f"), kind, path.name, path, job_id, item)

    def file(self, file_id):
        row = self._db.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return None if row is None else self._file(row)

    def files(self, kind=None, origin=None):
        """The file rows with the given kind and origin, newest first."""
        where, args = [], []
        if kind is not None:
            where.append("kind = ?")
            args.append(kind)
        if origin is not None:
            where.append("job_id IS NULL" if origin == "upload" else "job_id IS NOT NULL")
        return [self._file(row) for row in self._newest_first("files", where, args, -1)]

    def output_files(self, job_id):
        """The file rows of the outputs of the job, ordered by item, then by registration."""
        rows = self._db.execute("SELECT * FROM files WHERE job_id = ? ORDER BY item, seq", (job_id,))
        return [self._file(row) for row in rows]

    # Jobs.

    def job_dir(self, job_id):
        """The folder of the job; it exists from the creation of the job to its deletion."""
        return DATA / "jobs" / job_id

    def create_job(self, job, params, inputs, count, seed, priority, label, notify_url):
        """Inserts a queued job with the given fields and returns its row."""
        with self._db:
            job_id = self._insert_job(job, params, inputs, count, seed, priority, label, notify_url, None)
        return self.job(job_id)

    def job(self, job_id):
        row = self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else self._job(row)

    def jobs(self, status=None, job=None, batch_id=None, before=None, limit=50):
        """Up to `limit` job rows, newest first, with the given status, job and batch_id; those with a `seq` below
        `before` when it is given. The `seq` of the last row is the `before` of the next page."""
        where, args = [], []
        for column, value in (("status", status), ("job", job), ("batch_id", batch_id)):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        if before is not None:
            where.append("seq < ?")
            args.append(before)
        return [self._job(row) for row in self._newest_first("jobs", where, args, limit)]

    def start_job(self, job_id):
        """Marks the job running and sets its start time."""
        with self._db:
            self._db.execute("UPDATE jobs SET status = 'running', started = ? WHERE job_id = ?", (time.time(), job_id))

    def add_work(self, job_id, seconds, dollars):
        """Adds seconds of stage execution and cloud dollars to the job. It is the `charge` of the calls of the job:
        pass `functools.partial(store.add_work, job_id)`."""
        with self._db:
            self._db.execute("UPDATE jobs SET work_seconds = work_seconds + ?, cloud_dollars = cloud_dollars + ? "
                             "WHERE job_id = ?", (seconds, dollars, job_id))

    def finish_job(self, job_id, status, kind=None, message=None):
        """Sets the job's status (one of FINISHED) and finish time, and its error when `kind` is given."""
        with self._db:
            self._db.execute("UPDATE jobs SET status = ?, error_kind = ?, error_message = ?, finished = ? "
                             "WHERE job_id = ?", (status, kind, message, time.time(), job_id))

    def delete_job(self, job_id):
        """Deletes a finished job with its output rows and its folder. Raises Refused when the job is not finished, or
        when an output of it is an input of a queued or running job."""
        job = self.job(job_id)
        if job["status"] not in FINISHED:
            raise Refused(f"job {job_id} is {job['status']}; cancel it, then delete it")
        outputs = {file_id for item in job["outputs"] for file_id in item}
        rows = self._db.execute("SELECT job_id, inputs FROM jobs WHERE status IN ('queued', 'running')")
        users = [row["job_id"] for row in rows
                 if outputs & {file_id for ids in json.loads(row["inputs"]).values() for file_id in ids}]
        if users:
            raise Refused(f"outputs of job {job_id} are inputs of the unfinished jobs {', '.join(users)}; "
                          "wait for them or cancel them, then delete")
        with self._db:
            self._db.execute("DELETE FROM files WHERE job_id = ?", (job_id,))
            self._db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        shutil.rmtree(self.job_dir(job_id))

    def fail_unfinished(self):
        """Marks every queued or running job failed with the error `stopped` / "factory stopped" and returns their
        job_ids. The entry point calls it once at start."""
        with self._db:
            rows = self._db.execute("SELECT job_id FROM jobs WHERE status IN ('queued', 'running')").fetchall()
            self._db.execute("UPDATE jobs SET status = 'failed', error_kind = 'stopped', "
                             "error_message = 'factory stopped', finished = ? "
                             "WHERE status IN ('queued', 'running')", (time.time(),))
        return [row["job_id"] for row in rows]

    # Batches.

    def create_batch(self, label, notify_url, jobs):
        """Inserts a batch and its queued jobs in one transaction. `jobs` is a list of dicts with the keyword arguments
        of `create_job`. Returns the batch."""
        batch_id = _new_id("b")
        with self._db:
            self._db.execute("INSERT INTO batches VALUES (?, ?, ?, ?)", (batch_id, label, notify_url, time.time()))
            for fields in jobs:
                self._insert_job(**fields, batch_id=batch_id)
        return self.batch(batch_id)

    def batch(self, batch_id):
        """The batch as a dict with batch_id, label, notify_url, created and `jobs`: the rows of its jobs, oldest
        first."""
        row = self._db.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        rows = self._db.execute("SELECT * FROM jobs WHERE batch_id = ? ORDER BY seq", (batch_id,))
        return {**dict(row), "jobs": [self._job(job) for job in rows]}

    # Internals.

    def _insert_file(self, file_id, kind, name, path, job_id, item):
        with self._db:
            self._db.execute("INSERT INTO files (file_id, kind, name, path, size, created, job_id, item) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (file_id, kind, name, path.relative_to(DATA).as_posix(),
                              path.stat().st_size, time.time(), job_id, item))
        return self.file(file_id)

    def _insert_job(self, job, params, inputs, count, seed, priority, label, notify_url, batch_id):
        job_id = _new_id("j")
        self._db.execute("INSERT INTO jobs (job_id, job, params, inputs, count, seed, priority, label, notify_url, "
                         "batch_id, status, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
                         (job_id, job, json.dumps(params), json.dumps(inputs), count, seed, priority, label,
                          notify_url, batch_id, time.time()))
        self.job_dir(job_id).mkdir(parents=True)
        return job_id

    def _newest_first(self, table, where, args, limit):
        sql = f"SELECT * FROM {table}" + (" WHERE " + " AND ".join(where) if where else "")
        return self._db.execute(sql + " ORDER BY seq DESC LIMIT ?", (*args, limit)).fetchall()

    def _file(self, row):
        file = dict(row)
        del file["seq"]
        file["origin"] = "upload" if file["job_id"] is None else "output"
        file["path"] = str(DATA / file["path"])
        return file

    def _job(self, row):
        job = dict(row)
        job["params"] = json.loads(job["params"])
        job["inputs"] = json.loads(job["inputs"])
        kind, message = job.pop("error_kind"), job.pop("error_message")
        job["error"] = None if kind is None else {"kind": kind, "message": message}
        job["outputs"] = [[] for _ in range(job["count"])]
        for file in self.output_files(job["job_id"]):
            job["outputs"][file["item"]].append(file["file_id"])
        return job
