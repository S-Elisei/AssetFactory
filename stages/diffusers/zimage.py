"""Stage zimage: one image from Tongyi-MAI/Z-Image-Turbo, from a prompt (`image` is None) or from a prompt and a source
image. `width` and `height` are used only without `image`, `strength` only with `image`; the unused ones are passed as
None."""
import shutil
from pathlib import Path

import diffusers_common as common
import mapped
import torch
from context import MODELS, InputError
from diffusers import AutoencoderKL, ZImageImg2ImgPipeline, ZImagePipeline, ZImageTransformer2DModel
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open

REPO = "Tongyi-MAI/Z-Image-Turbo"
KEEP_LOADED = False
FILES = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]
# Folder that download() writes and load() reads: the transformer shards of the checkpoint converted to bf16.
BF16_TRANSFORMER = MODELS / "z-image-turbo" / "transformer-bf16"
# Range in pixels of the long side of the output of an image-to-image run.
MIN_SIDE = 512
MAX_SIDE = 2048


def download():
    """Fetches the checkpoint and writes BF16_TRANSFORMER; shards already converted are kept."""
    source = Path(snapshot_download(REPO, allow_patterns=FILES)) / "transformer"
    BF16_TRANSFORMER.mkdir(parents=True, exist_ok=True)
    for shard in sorted(source.glob("*.safetensors")):
        target = BF16_TRANSFORMER / shard.name
        if not target.exists():
            with safe_open(shard, "pt") as file:
                tensors = {key: file.get_tensor(key).to(torch.bfloat16) for key in file.keys()}
            mapped.save_weights(target, tensors)
    for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
        shutil.copyfile(source / name, BF16_TRANSFORMER / name)


def load():
    """Returns {"text_to_image": pipeline, "image_to_image": pipeline, "groups": transformer groups, "prompt_cache":
    dict}; both pipelines share one set of components."""
    snapshot = snapshot_download(REPO, allow_patterns=FILES)
    text_encoder = common.load_text_encoder(snapshot)
    transformer = ZImageTransformer2DModel.from_pretrained(BF16_TRANSFORMER, dtype=torch.bfloat16)
    groups = common.stream_transformer(transformer)
    vae = common.load_vae(AutoencoderKL, snapshot)
    text_to_image = ZImagePipeline.from_pretrained(snapshot, transformer=transformer, text_encoder=text_encoder,
                                                   vae=vae)
    image_to_image = ZImageImg2ImgPipeline(**text_to_image.components)
    text_to_image.set_progress_bar_config(disable=True)
    image_to_image.set_progress_bar_config(disable=True)
    return {"text_to_image": text_to_image, "image_to_image": image_to_image, "groups": groups, "prompt_cache": {}}


def _source_image(path):
    """Opens the image file at `path` as RGB and resizes it, keeping its aspect ratio, so that its long side lies in
    [MIN_SIDE, MAX_SIDE] and each side is rounded to a multiple of 16. Returns the image."""
    try:
        source = Image.open(path).convert("RGB")
    except OSError:
        raise InputError("image: the file is not a readable image; send a PNG, JPEG or WEBP")
    scale = min(max(max(source.size), MIN_SIDE), MAX_SIDE) / max(source.size)
    size = tuple(round(side * scale / 16) * 16 for side in source.size)
    return source if size == source.size else source.resize(size, Image.Resampling.LANCZOS)


def run(ctx, prompt, steps, seed, width, height, image, strength):
    if image is None:
        pipeline = ctx.model["text_to_image"]
        inputs = {"width": width, "height": height}
    else:
        pipeline = ctx.model["image_to_image"]
        source = _source_image(image)
        inputs = {"image": source, "strength": strength}
        width, height = source.size

    embeds = common.prompt_embeds(ctx, ctx.model["prompt_cache"], pipeline, prompt, do_classifier_free_guidance=False)
    common.keep_resident(ctx.model["groups"], common.resident_budget(width * height))
    result = pipeline(
        prompt_embeds=embeds,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=torch.Generator(common.DEVICE).manual_seed(seed),
        callback_on_step_end=common.sampling_callback(ctx),
        **inputs,
    ).images[0]

    ctx.progress(0.97, "saving")
    path = ctx.dir / "image.png"
    result.save(path)
    return {"image": str(path), "seed": seed, "width": result.width, "height": result.height, "steps": steps}
