"""Stage stableaudio: one sound effect from Stable Audio 3 Small-SFX, written as `audio.wav` (16-bit PCM at the model's
sample rate). Without `audio` it is generated from `prompt` and `duration`. With `audio`: `noise_level` gives a new
version of the clip, as long as the clip; `start_seconds` and `end_seconds` regenerate that region of the clip; a
`duration` alone continues the clip to `duration` seconds. The parameters of the tasks that are not run are passed as
None."""
import json
from pathlib import Path

import audio_common as common
import hub
import mapped
import soundfile
import torch
from accelerate import init_empty_weights
from context import MODELS, InputError
from safetensors import safe_open
from stable_audio_3 import StableAudioModel
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.models.conditioners import T5GemmaConditioner

REPO = "stabilityai/stable-audio-3-small-sfx"
REVISION = "ae12755283df9d62ca39a9b050a39a0b607b8c20"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 3.5
FILES = ["model_config.json", "model.safetensors", "t5gemma-b-b-ul2/config.json", "t5gemma-b-b-ul2/model.safetensors",
         "t5gemma-b-b-ul2/tokenizer.json", "t5gemma-b-b-ul2/tokenizer_config.json",
         "t5gemma-b-b-ul2/special_tokens_map.json"]
# File that download() writes and load() reads: model.safetensors in fp16.
FP16_WEIGHTS = MODELS / "stableaudio" / "model.safetensors"
DEVICE = torch.device("cuda")
FORMATS = "WAV, FLAC or OGG"


def download():
    """Fetches the checkpoint and writes FP16_WEIGHTS; an existing FP16_WEIGHTS is kept."""
    source = Path(hub.snapshot(REPO, REVISION, FILES)) / "model.safetensors"
    if not FP16_WEIGHTS.exists():
        FP16_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(source, "pt") as file:
            tensors = {key: file.get_tensor(key).to(torch.float16) for key in file.keys()}
        mapped.save_weights(FP16_WEIGHTS, tensors)


def load():
    """Returns the StableAudioModel. Everything but the prompt encoder is built without weights, loaded from FP16_WEIGHTS
    onto the GPU and runs in fp16. The prompt encoder (T5Gemma) is built by transformers directly on the GPU and runs in
    bf16, from the folder of the pinned snapshot that the config names."""
    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    config = json.loads((snapshot / "model_config.json").read_text(encoding="utf-8"))
    conditioning = config["model"]["conditioning"]
    prompt = next(item for item in conditioning["configs"] if item["id"] == "prompt")
    others = {**conditioning, "configs": [item for item in conditioning["configs"] if item is not prompt]}
    with init_empty_weights():
        model = create_diffusion_cond_from_config({**config, "model": {**config["model"], "conditioning": others}})
    with torch.device(DEVICE):
        model.conditioner.conditioners["prompt"] = T5GemmaConditioner(
            output_dim=conditioning["cond_dim"],
            **{**prompt["config"], "model_path": str(snapshot / prompt["config"]["subfolder"]), "subfolder": None})
    with safe_open(FP16_WEIGHTS, "pt", device="cuda") as file:
        model.load_state_dict({key: file.get_tensor(key) for key in file.keys()}, assign=True)
    model.to(DEVICE, torch.float16).eval().requires_grad_(False)
    return StableAudioModel(model, config, "cuda", True)


def _on_step(ctx, steps):
    """Returns the sampler callback: it raises Cancelled when the run was cancelled and reports the finished sampling
    step and, after the last step, the start of decoding."""

    def callback(state):
        ctx.check_cancel()
        step = state["i"] + 1
        ctx.progress(0.9 * step / steps, f"sampling step {step}/{steps}")
        if step == steps:
            ctx.progress(0.9, "decoding")

    return callback


def run(ctx, prompt, steps, seed, duration, audio, noise_level, start_seconds, end_seconds):
    model = ctx.model
    length, options = duration, {}
    if audio is not None:
        with common.open_audio("audio", audio, FORMATS) as file:
            seconds = file.frames / file.samplerate
            clip = (file.samplerate, torch.from_numpy(file.read(dtype="float32", always_2d=True).T.copy()))
        longest = model.model_config["sample_size"] / model.model.sample_rate
        if noise_level is not None:
            length = min(seconds, longest)
            options = {"init_audio": clip, "init_noise_level": noise_level}
        elif start_seconds is not None:
            length = min(seconds, longest)
            end = min(end_seconds, length)
            if end <= start_seconds:
                raise InputError(f"start_seconds, end_seconds: the region from {start_seconds} s to {end_seconds} s is "
                                 f"empty within the source of {length:.2f} s; send start_seconds below end_seconds and "
                                 f"below {length:.2f}")
            options = {"inpaint_audio": clip, "inpaint_mask_start_seconds": start_seconds,
                       "inpaint_mask_end_seconds": end}
        else:
            if duration <= seconds:
                raise InputError(f"duration: {duration} s does not exceed the source of {seconds:.2f} s; send a duration "
                                 f"above {seconds:.2f}")
            options = {"inpaint_audio": clip, "inpaint_mask_start_seconds": seconds, "inpaint_mask_end_seconds": duration}

    ctx.progress(0.0, "preparing")
    samples = model.generate(prompt=prompt, duration=length, steps=steps, seed=seed, callback=_on_step(ctx, steps),
                             **options)[0].cpu().numpy().T

    ctx.progress(0.97, "writing")
    rate = model.model.sample_rate
    path = ctx.dir / "audio.wav"
    soundfile.write(path, samples, rate, subtype="PCM_16")
    return {"audio": str(path), "duration": len(samples) / rate}
