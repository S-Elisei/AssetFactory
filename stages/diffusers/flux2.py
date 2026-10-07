"""Stage flux2: one image from black-forest-labs/FLUX.2-klein-4B, from a prompt (`reference_images` empty) or from a
prompt and reference image paths to edit or combine."""
import diffusers_common as common
import torch
from context import InputError
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from huggingface_hub import snapshot_download
from PIL import Image

REPO = "black-forest-labs/FLUX.2-klein-4B"
KEEP_LOADED = False
# Text-encoder layers whose hidden states make up the prompt embeddings; the text encoder keeps max + 1 decoder layers.
TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)
FILES = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]


def download():
    snapshot_download(REPO, allow_patterns=FILES)


def load():
    """Returns {"pipeline": pipeline, "groups": transformer groups, "prompt_cache": dict}."""
    snapshot = snapshot_download(REPO, allow_patterns=FILES)
    text_encoder = common.load_text_encoder(snapshot, max(TEXT_ENCODER_OUT_LAYERS) + 1)
    transformer = Flux2Transformer2DModel.from_pretrained(snapshot, subfolder="transformer", dtype=torch.bfloat16)
    groups = common.stream_transformer(transformer)
    vae = common.load_vae(AutoencoderKLFlux2, snapshot)
    pipeline = Flux2KleinPipeline.from_pretrained(snapshot, transformer=transformer, text_encoder=text_encoder, vae=vae)
    pipeline.set_progress_bar_config(disable=True)
    return {"pipeline": pipeline, "groups": groups, "prompt_cache": {}}


def _reference(number, path):
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        raise InputError(f"reference_images[{number}]: the file is not a readable image; send a PNG, JPEG or WEBP")


def run(ctx, prompt, width, height, steps, seed, reference_images):
    references = [_reference(number, path) for number, path in enumerate(reference_images)]

    pipeline = ctx.model["pipeline"]
    embeds = common.prompt_embeds(ctx, ctx.model["prompt_cache"], pipeline, prompt,
                                   text_encoder_out_layers=TEXT_ENCODER_OUT_LAYERS)
    common.keep_resident(ctx.model["groups"], 0 if references else common.resident_budget(width * height))
    result = pipeline(
        prompt_embeds=embeds,
        image=references or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=1.0,
        generator=torch.Generator(common.DEVICE).manual_seed(seed),
        callback_on_step_end=common.sampling_callback(ctx),
    ).images[0]

    ctx.progress(0.97, "saving")
    path = ctx.dir / "image.png"
    result.save(path)
    return {"image": str(path), "seed": seed, "width": result.width, "height": result.height, "steps": steps}
