"""Entry point of the factory. It binds HOST:PORT first, so that a start with the port taken ends before it touches the
Store. It then marks the jobs that `Store` holds as queued or running as failed with "factory stopped", sends their
notifications, and serves the Api and the UI with one uvicorn server in one process; the asyncio loop is the Proactor
loop of Windows, which runs the subprocesses of the Workers.

Ctrl+C stops the server, then the Runner, then the LocalQueue and the CloudQueue together, then the Notifier. A further
Ctrl+C during this sequence changes nothing."""
import asyncio
import socket
from contextlib import asynccontextmanager, suppress

import uvicorn

from core.api import Api
from core.cloud_queue import CloudQueue
from core.local_queue import LocalQueue
from core.notifier import Notifier
from core.runner import Runner
from core.store import Store

HOST = "127.0.0.1"
PORT = 8700


async def serve():
    sock = socket.socket()
    sock.bind((HOST, PORT))
    store = Store()
    notifier = Notifier(store)
    notifier.jobs_stopped(store.fail_unfinished())
    local_queue, cloud_queue = LocalQueue(), CloudQueue()
    runner = Runner(store, notifier, local_queue, cloud_queue)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await runner.stop()
        await asyncio.gather(local_queue.stop(), cloud_queue.stop())
        await notifier.stop()

    api = Api(store, runner, local_queue, cloud_queue, lifespan)
    config = uvicorn.Config(api.app, host=HOST, port=PORT, access_log=False, timeout_graceful_shutdown=0)
    await uvicorn.Server(config).serve(sockets=[sock])


with suppress(KeyboardInterrupt):
    asyncio.run(serve())
