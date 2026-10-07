"""Code shared by the stages of the audio environments: seeding and opening input audio files. Imports numpy, soundfile,
torch and the Worker's `context`."""
import random

import numpy as np
import soundfile
import torch
from context import InputError


def seed_everything(seed):
    """Seeds the random generators of Python, numpy and torch with `seed`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def open_audio(name, path, formats):
    """Returns the soundfile.SoundFile of the audio file `path`, which the caller closes. Raises InputError naming the
    input `name` and the `formats` to send when the file cannot be read."""
    try:
        return soundfile.SoundFile(path)
    except soundfile.LibsndfileError:
        raise InputError(f"{name}: the file is not a readable audio file; send {formats}")
