"""Stage chatterbox: one line of speech from Chatterbox Multilingual V3, written as `speech.wav` (16-bit PCM mono at the
model's sample rate). The voice is the one cloned from `voice_reference`, or the built-in voice when it is None."""
import os

from context import MODELS

# Folder of the Chinese segmenter model that the tokenizer loads; read when spacy_pkuseg is first imported.
PKUSEG = MODELS / "pkuseg"
os.environ["PKUSEG_HOME"] = str(PKUSEG)

from pathlib import Path

import audio_common as common
import hub
import soundfile
import torch
from accelerate import init_empty_weights
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import S3_TOKEN_RATE
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from safetensors import safe_open

REPO = "ResembleAI/chatterbox"
REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 3.5
T3_FILE = "t3_mtl23ls_v3.safetensors"
FILES = ["ve.pt", T3_FILE, "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt"]
# Segmenter model of spacy_pkuseg that the tokenizer loads. Documented: the default of `pkuseg()`.
SEGMENTER = "spacy_ontonotes"
DEVICE = torch.device("cuda")
# Speech tokens that one run generates at most. Documented.
MAX_TOKENS = 1000


def download():
    """Fetches the V3 checkpoint files, the Cangjie table at REVISION in the cache folder that the tokenizer reads it
    from, with the folder's `main` ref set to REVISION, and the segmenter model."""
    from spacy_pkuseg.config import config
    from spacy_pkuseg.download import download_model

    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    hub.file(REPO, REVISION, "Cangjie5_TC.json", cache_dir=snapshot)
    ref = snapshot / "models--ResembleAI--chatterbox" / "refs" / "main"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(REVISION)
    download_model(config.model_urls[SEGMENTER], config.pkuseg_home, config.model_hash[SEGMENTER])


def _fill(module, state):
    """Loads `state` (tensors on the GPU) into `module`, built without weights, and returns `module` on the GPU in eval
    mode."""
    module.load_state_dict(state, assign=True)
    return module.to(DEVICE).eval()


def load():
    """Returns {"model": the ChatterboxMultilingualTTS, "builtin": the Conditionals of the built-in voice}. The three
    networks are built without weights and take their tensors from the files straight onto the GPU, in fp32."""
    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    with init_empty_weights():
        voice_encoder, t3, s3gen = VoiceEncoder(), T3(T3Config.multilingual()), S3Gen()
    voice_encoder = _fill(voice_encoder, torch.load(snapshot / "ve.pt", map_location=DEVICE, mmap=True,
                                                    weights_only=True))
    with safe_open(snapshot / T3_FILE, "pt", device="cuda") as file:
        t3 = _fill(t3, {key: file.get_tensor(key) for key in file.keys()})
    s3gen = _fill(s3gen, torch.load(snapshot / "s3gen.pt", map_location=DEVICE, mmap=True, weights_only=True))
    tokenizer = MTLTokenizer(str(snapshot / "grapheme_mtl_merged_expanded_v1.json"))
    builtin = Conditionals.load(snapshot / "conds.pt", map_location=DEVICE)
    return {"model": ChatterboxMultilingualTTS(t3, s3gen, voice_encoder, tokenizer, "cuda", conds=builtin),
            "builtin": builtin}


def run(ctx, text, language, exaggeration, cfg_weight, temperature, repetition_penalty, min_p, top_p, voice_reference,
        seed):
    model = ctx.model["model"]
    if voice_reference is None:
        model.conds = ctx.model["builtin"]
    else:
        common.open_audio("voice_reference", voice_reference).close()
    common.seed_everything(seed)
    tokens = 0

    def on_token(*_):
        nonlocal tokens
        ctx.check_cancel()
        tokens += 1
        ctx.progress(0.05 + 0.85 * tokens / MAX_TOKENS, f"{tokens / S3_TOKEN_RATE:.1f} s of audio")

    ctx.progress(0.05, "synthesizing")
    hook = model.t3.tfmr.register_forward_pre_hook(on_token)
    try:
        wave = model.generate(text, language_id=language, audio_prompt_path=voice_reference, exaggeration=exaggeration,
                              cfg_weight=cfg_weight, temperature=temperature, repetition_penalty=repetition_penalty,
                              min_p=min_p, top_p=top_p)[0].numpy()
    finally:
        hook.remove()

    ctx.progress(0.95, "writing")
    path = ctx.dir / "speech.wav"
    soundfile.write(path, wave, model.sr, subtype="PCM_16")
    return {"audio": str(path), "duration": len(wave) / model.sr}
