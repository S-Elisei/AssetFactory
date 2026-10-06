"""Names shared by the Worker and the stage modules. Standard library only."""
import time

# Minimum seconds between two progress messages of a run that have the same text.
PROGRESS_INTERVAL = 0.25


class InputError(Exception):
    """An unusable input. The message names the state and the next action, written for whoever sent the input."""


class Cancelled(Exception):
    """The run was cancelled."""


class Context:
    """Passed to the `run` of a stage. `dir` is the folder for the output files; `model` is the result of the stage's
    `load`, None for a stage without `load`."""

    def __init__(self, run_id, directory, send, cancelled):
        self.id = run_id
        self.dir = directory
        self.model = None
        self._send = send
        self._cancelled = cancelled
        self._last_progress = float("-inf")
        self._last_message = None

    def progress(self, fraction, message):
        """Reports `fraction` (0..1 of this run) and `message`. A message whose text differs from the previous message
        of the run is always sent; a message with the same text is sent at most once per PROGRESS_INTERVAL."""
        now = time.monotonic()
        if message != self._last_message or now - self._last_progress >= PROGRESS_INTERVAL:
            self._last_progress = now
            self._last_message = message
            self._send({"type": "progress", "id": self.id, "fraction": fraction, "message": message})

    def check_cancel(self):
        """Raises Cancelled when this run was cancelled."""
        if self.id in self._cancelled:
            raise Cancelled("cancelled")
