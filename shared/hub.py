"""Fetches Hugging Face repository files at a pinned revision: from the cache without the network when the revision is
cached, from the Hub otherwise."""
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def snapshot(repo, revision, files=None, **options):
    """Returns the path of the snapshot of `repo` at `revision` holding the files that match the patterns `files`."""
    try:
        return snapshot_download(repo, revision=revision, allow_patterns=files, local_files_only=True, **options)
    except LocalEntryNotFoundError:
        return snapshot_download(repo, revision=revision, allow_patterns=files, **options)


def file(repo, revision, name, **options):
    """Returns the path of the file `name` of `repo` at `revision`."""
    try:
        return hf_hub_download(repo, name, revision=revision, local_files_only=True, **options)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, name, revision=revision, **options)
