"""Exceptions of a stage call, raised by `LocalQueue.call` and `CloudQueue.call`. An unusable input raises `InputError`
of `worker/context.py` (importable once `core.layout` is imported) with the message of the Worker or of the container."""

# The message of the StageCancelled that a queue raises for a call it ends when it is stopped.
STOPPING = "cancelled: the factory is stopping"


class OutOfMemory(Exception):
    """The stage ran out of GPU or host memory. The message is the library's."""


class StageFailed(Exception):
    """The stage failed. The message names the exception of the stage, or the exit of the Worker, or the failed Modal
    call."""


class StageCancelled(Exception):
    """The stage ended as cancelled."""
