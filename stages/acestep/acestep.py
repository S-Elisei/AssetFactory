"""Stage acestep: one piece of music from ACE-Step 1.5 (turbo DiT, optional 5Hz LM planner), written as `music.wav`
(16-bit PCM stereo at the model's sample rate). Without `audio` it is generated from
`caption`, `lyrics` and `duration`; `thinking` runs the LM planner first. With `audio`: `start_seconds` and `end_seconds`
repaint that range of the track, `extend_seconds` appends music to it, `cover_strength` re-renders it in the style of the
caption. The parameters of the tasks that are not run are passed as None.

The DiT, the VAE, the text encoder and the LM are built without weights and take their parameters from memory maps of
their files. The DiT stays on the GPU between runs; the VAE and the text encoder are on the GPU only while the model
code uses them; the LM, built on the first run that uses it, is on the GPU only while it plans, and the DiT is back in
its memory maps meanwhile."""
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import audio_common as common
import hub
import mapped
import soundfile
import torch
from accelerate import init_empty_weights
from acestep.constants import DURATION_MAX
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler
from acestep.models.common.configuration_acestep_v15 import AceStepConfig
from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
from context import MODELS, InputError
from diffusers import AutoencoderOobleck
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO = "ACE-Step/Ace-Step1.5"
REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 5.0
# Folder of the checkpoint files, laid out as in the repository.
CHECKPOINTS = MODELS / "acestep"
DIT = CHECKPOINTS / "acestep-v15-turbo"
VAE = CHECKPOINTS / "vae"
TEXT_ENCODER = CHECKPOINTS / "Qwen3-Embedding-0.6B"
LM = CHECKPOINTS / "acestep-5Hz-lm-1.7B"
# The files `load` and the LM build read.
FILES = ["acestep-v15-turbo/config.json", "acestep-v15-turbo/model.safetensors", "acestep-v15-turbo/silence_latent.pt",
         "vae/*", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*"]
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
# Fractions of a run's progress at which the diffusion steps start and end. Documented.
DIFFUSION_START, DIFFUSION_END = 0.52, 0.79


def download():
    hub.snapshot(REPO, REVISION, FILES, local_dir=CHECKPOINTS)


def _attach(model, tensors):
    """Takes the parameters of `model`, built without weights, from `tensors` (memory maps), puts its buffers on the GPU,
    converts the floating ones to DTYPE, and returns `model` in eval mode."""
    mapped.attach(model, tensors)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(DEVICE, DTYPE) if buffer.is_floating_point() else buffer.data.to(DEVICE)
    return model.eval()


def _lm(path):
    """Returns the LM of the folder `path` (a Qwen3Model checkpoint) as a causal LM whose output projection is tied to
    its input embeddings."""
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), dtype=DTYPE)
    tensors = {"model." + key: tensor for key, tensor in mapped.map_tensors([Path(path) / "model.safetensors"]).items()}
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    model = _attach(model, tensors)
    model.tie_weights()
    return model


def _model_context(dit):
    """Returns the replacement of the handler's `_load_model_context`: the DiT is put on the GPU for good when it is in
    its memory maps; the VAE and the text encoder are on the GPU for the duration of the block."""

    @contextmanager
    def context(name):
        if name == "model":
            if next(dit.model.parameters()).device.type == "cpu":
                mapped.to_device(dit.model, DEVICE)
            yield
        else:
            with mapped.on_gpu(DEVICE, getattr(dit, name)):
                yield

    return context


def load():
    """Returns {"dit": the AceStepHandler, "llm": None}; run() puts the LMHandler into "llm" when a run plans with it."""
    os.environ["ACESTEP_PROJECT_ROOT"] = str(MODELS.parent)
    dit = AceStepHandler()
    dit.device, dit.dtype, dit.offload_to_cpu = "cuda", DTYPE, True
    config = AceStepConfig.from_pretrained(DIT)
    with init_empty_weights():
        model = AceStepConditionGenerationModel._from_config(config, dtype=DTYPE, attn_implementation="sdpa")
    dit.model = _attach(model, mapped.map_tensors([DIT / "model.safetensors"]))
    dit.config = model.config
    mapped.to_device(model, DEVICE)
    with init_empty_weights():
        vae = AutoencoderOobleck.from_config(AutoencoderOobleck.load_config(VAE))
    dit.vae = _attach(vae, mapped.map_tensors([VAE / "diffusion_pytorch_model.safetensors"]))
    with init_empty_weights():
        text_encoder = AutoModel.from_config(AutoConfig.from_pretrained(TEXT_ENCODER), dtype=DTYPE)
    dit.text_encoder = _attach(text_encoder, mapped.map_tensors([TEXT_ENCODER / "model.safetensors"]))
    dit.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    dit.silence_latent = torch.load(DIT / "silence_latent.pt", weights_only=True, map_location=DEVICE).transpose(
        1, 2).to(DTYPE)
    dit._load_model_context = _model_context(dit)
    return {"dit": dit, "llm": None}


def _llm(ctx):
    """Returns the LMHandler of the run's model, building it on first use. Its planning moves the DiT to its memory maps
    and the LM to the GPU for its duration."""
    model = ctx.model
    if model["llm"] is None:
        ctx.progress(0.0, "loading the LM")
        llm = LLMHandler()

        def load_lm(path, device):
            llm.llm, llm.llm_backend, llm.llm_initialized = _lm(path), "pt", True
            return True, ""

        llm._load_pytorch_model = load_lm
        status, loaded = llm.initialize(checkpoint_dir=str(CHECKPOINTS), lm_model_path=LM.name, backend="pt",
                                        device="cuda", offload_to_cpu=False)
        if not loaded:
            raise RuntimeError(status)
        plan = llm.generate_with_stop_condition

        def plan_on_gpu(*args, **kwargs):
            mapped.to_host(model["dit"].model)
            with mapped.on_gpu(DEVICE, llm.llm):
                return plan(*args, **kwargs)

        llm.generate_with_stop_condition = plan_on_gpu
        model["llm"] = llm
    return model["llm"]


def _step_hook(ctx, steps):
    """Returns a forward hook for the DiT decoder that counts each call as one of the `steps` diffusion steps run: it
    raises Cancelled when the run was cancelled and reports the finished step."""
    step = 0

    def hook(*_):
        nonlocal step
        step += 1
        ctx.check_cancel()
        ctx.progress(DIFFUSION_START + (DIFFUSION_END - DIFFUSION_START) * step / steps, f"diffusion step {step}/{steps}")

    return hook


def _progress(ctx):
    """Returns the progress callback for the model code; it checks for cancellation and reports only when called from the
    Worker's thread."""

    def progress(value, desc=None, *_, **__):
        if threading.current_thread() is threading.main_thread():
            ctx.check_cancel()
            ctx.progress(min(value, 0.95), desc)

    return progress


def run(ctx, caption, lyrics, duration, bpm, keyscale, timesignature, vocal_language, thinking, rewrite_caption,
        lm_temperature, steps, shift, start_seconds, end_seconds, extend_seconds, cover_strength, audio, seed):
    options = {"caption": caption, "lyrics": lyrics.strip() or "[Instrumental]", "vocal_language": vocal_language,
               "inference_steps": steps, "shift": shift, "seed": seed, "thinking": False, "use_cot_metas": False,
               "use_cot_caption": False, "use_cot_language": False}
    llm = None
    if audio is None:
        options.update(duration=duration, bpm=bpm or None, keyscale=keyscale, timesignature=timesignature)
        if thinking:
            options.update(thinking=True, use_cot_metas=True, use_cot_caption=rewrite_caption, use_cot_language=True,
                           lm_temperature=lm_temperature)
            llm = _llm(ctx)
    else:
        with common.open_audio("audio", audio) as file:
            source = file.frames / file.samplerate
        if source > DURATION_MAX:
            raise InputError(f"audio: the source is {source:.1f} s long; send a source of at most {DURATION_MAX} s")
        options["src_audio"] = audio
        if start_seconds is not None:
            end = source if end_seconds < 0 else end_seconds
            if end <= start_seconds or start_seconds >= source:
                raise InputError(f"start_seconds, end_seconds: the range from {start_seconds} s to {end:.2f} s is empty "
                                 f"within the source of {source:.2f} s; send start_seconds below end_seconds and below "
                                 f"{source:.2f}")
            options.update(task_type="repaint", repainting_start=start_seconds, repainting_end=end,
                           chunk_mask_mode="explicit")
        elif extend_seconds is not None:
            if source + extend_seconds > DURATION_MAX:
                raise InputError(f"extend_seconds: the source of {source:.1f} s plus {extend_seconds} s exceeds "
                                 f"{DURATION_MAX} s; send extend_seconds of at most {DURATION_MAX - source:.1f}")
            options.update(task_type="repaint", repainting_start=source, repainting_end=source + extend_seconds,
                           chunk_mask_mode="explicit")
        else:
            options.update(task_type="cover", audio_cover_strength=cover_strength)

    dit = ctx.model["dit"]
    hooks = [dit.model.decoder.register_forward_hook(_step_hook(ctx, steps))]
    if llm is not None:
        hooks.append(llm.llm.register_forward_pre_hook(lambda *_: ctx.check_cancel()))
    common.seed_everything(seed)
    ctx.progress(0.0, "preparing")
    try:
        result = generate_music(dit, llm, GenerationParams(**options),
                                GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed]),
                                progress=_progress(ctx))
    finally:
        for hook in hooks:
            hook.remove()
    if not result.success:
        ctx.check_cancel()
        raise RuntimeError(result.error)

    sample = result.audios[0]
    rate = sample["sample_rate"]
    wave = sample["tensor"].numpy().T
    if audio is None:
        wave = wave[:round(duration * rate)]
    ctx.progress(0.97, "writing")
    path = ctx.dir / "music.wav"
    soundfile.write(path, wave, rate, subtype="PCM_16")
    details = {"audio": str(path), "duration": len(wave) / rate}
    if audio is None:
        used = sample["params"]
        details.update(bpm=used["bpm"] or used["cot_bpm"], keyscale=used["keyscale"] or used["cot_keyscale"],
                       timesignature=used["timesignature"] or used["cot_timesignature"])
        if thinking and rewrite_caption:
            details["lm_caption"] = result.extra_outputs["lm_metadata"].get("caption")
    return details
