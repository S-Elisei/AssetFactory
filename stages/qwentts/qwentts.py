"""Stage qwentts: one line of speech from Qwen3-TTS 1.7B VoiceDesign in a voice described by `voice_description`, written
as `speech.wav` (16-bit PCM mono at the speech tokenizer's sample rate)."""
from pathlib import Path

import audio_common as common
import hub
import mapped
import soundfile
import torch
from accelerate import init_empty_weights
from context import MODELS
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from qwen_tts.core import Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from safetensors import safe_open

REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
REVISION = "5ecdb67327fd37bb2e042aab12ff7391903235d3"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 3.5
FILES = ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "vocab.json",
         "merges.txt", "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors"]
# File that download() writes and load() reads: the decoder tensors of speech_tokenizer/model.safetensors in bf16.
DECODER_WEIGHTS = MODELS / "qwentts" / "speech_decoder.safetensors"
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
# Generated speech frames per second of audio. Documented.
FRAME_RATE = 12.5
# Language names that the model takes for each language code of the run.
LANGUAGES = {"auto": "auto", "zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german",
             "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}


def download():
    """Fetches the checkpoint files and writes DECODER_WEIGHTS; an existing DECODER_WEIGHTS is kept."""
    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    if not DECODER_WEIGHTS.exists():
        DECODER_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(snapshot / "speech_tokenizer" / "model.safetensors", "pt") as file:
            tensors = {key: file.get_tensor(key).to(DTYPE) for key in file.keys() if key.startswith("decoder.")}
        mapped.save_weights(DECODER_WEIGHTS, tensors)


def _fill(module, path):
    """Loads the safetensors file `path` straight onto the GPU into `module`, built without weights, and returns `module`
    on the GPU in eval mode."""
    with safe_open(path, "pt", device="cuda") as file:
        module.load_state_dict({key: file.get_tensor(key) for key in file.keys()}, assign=True)
    return module.to(DEVICE).eval()


def load():
    """Returns the Qwen3TTSModel. The talker and the speech decoder are built without weights and take their tensors from
    the files straight onto the GPU, in bf16; the speech tokenizer's encoder is not built."""
    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    config = Qwen3TTSConfig.from_pretrained(snapshot)
    with init_empty_weights():
        talker = Qwen3TTSForConditionalGeneration._from_config(config, dtype=DTYPE, attn_implementation="sdpa")
    talker = _fill(talker, snapshot / "model.safetensors")

    decoder_config = Qwen3TTSTokenizerV2Config.from_pretrained(snapshot / "speech_tokenizer")
    with init_empty_weights():
        decoder = Qwen3TTSTokenizerV2Model._from_config(decoder_config, dtype=DTYPE)
    decoder.encoder = None
    tokenizer = Qwen3TTSTokenizer()
    tokenizer.model, tokenizer.config, tokenizer.device = _fill(decoder, DECODER_WEIGHTS), decoder_config, DEVICE
    talker.load_speech_tokenizer(tokenizer)
    return Qwen3TTSModel(talker, Qwen3TTSProcessor.from_pretrained(snapshot, fix_mistral_regex=True))


def run(ctx, text, voice_description, language, do_sample, temperature, top_k, top_p, repetition_penalty,
        subtalker_dosample, subtalker_temperature, subtalker_top_k, subtalker_top_p, max_new_tokens, seed):
    frames = 0

    def on_frame(*_):
        nonlocal frames
        ctx.check_cancel()
        frames += 1
        ctx.progress(0.05 + 0.85 * frames / max_new_tokens, f"{frames / FRAME_RATE:.1f} s of audio")

    hook = ctx.model.model.talker.register_forward_pre_hook(on_frame)
    common.seed_everything(seed)
    ctx.progress(0.02, "synthesizing")
    try:
        waves, rate = ctx.model.generate_voice_design(
            text=text, instruct=voice_description, language=LANGUAGES[language], do_sample=do_sample,
            temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            subtalker_dosample=subtalker_dosample, subtalker_temperature=subtalker_temperature,
            subtalker_top_k=subtalker_top_k, subtalker_top_p=subtalker_top_p, max_new_tokens=max_new_tokens)
    finally:
        hook.remove()

    ctx.progress(0.95, "writing")
    path = ctx.dir / "speech.wav"
    soundfile.write(path, waves[0], rate, subtype="PCM_16")
    return {"audio": str(path), "duration": len(waves[0]) / rate}
