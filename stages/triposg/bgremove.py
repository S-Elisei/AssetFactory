"""Stage bgremove: RGBA PNGs of the input images with the background removed by BiRefNet. An image with an alpha channel
on which at least CUTOUT_FRACTION of the pixels are at alpha 0 is used as is. The model's weights stay in memory maps of
the checkpoint file; `run` copies them to the GPU for its duration. `load`, `load_on_gpu` and `remove_background` do not
use the Worker context."""
from pathlib import Path

import hub
import mapped
import numpy as np
import torch
from accelerate import init_empty_weights
from context import InputError
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageSegmentation

REPO = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
KEEP_LOADED = True
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 3.0
# The files `load` reads: the model weights and its remote code.
FILES = ["config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"]
# Side in pixels of the square model input. Documented.
SIZE = 1024
# Share of fully transparent pixels from which an image counts as already cut out. Guessed.
CUTOUT_FRACTION = 0.05

# Documented: resize to the model input size and ImageNet normalisation.
PREPARE = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def download():
    hub.snapshot(REPO, REVISION, FILES)


def load():
    """Returns BiRefNet in fp16, built without weights, its parameters and buffers pointing to memory maps of the
    checkpoint file."""
    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForImageSegmentation.from_config(config, trust_remote_code=True)
    mapped.attach(model, mapped.map_tensors([snapshot / "model.safetensors"]))
    return model.eval()


def load_on_gpu():
    """Returns the model of `load` on the GPU for good: its tensors are copied there and the memory maps are released."""
    model = load().to("cuda")
    for tensor in (*model.parameters(), *model.buffers()):
        if hasattr(tensor, "host"):
            del tensor.host
    return model


def remove_background(model, image):
    """Returns the RGBA version of the PIL `image`: the image itself when at least CUTOUT_FRACTION of its pixels are at
    alpha 0, otherwise the image with the BiRefNet foreground mask as alpha."""
    rgba = image.convert("RGBA")
    if (np.asarray(rgba.getchannel("A")) == 0).mean() >= CUTOUT_FRACTION:
        return rgba
    batch = PREPARE(image.convert("RGB")).unsqueeze(0).to("cuda", torch.float16)
    with torch.no_grad():
        mask = model(batch)[-1].sigmoid()[0, 0].float().cpu().numpy()
    rgba.putalpha(Image.fromarray((mask * 255).round().astype(np.uint8)).resize(rgba.size, Image.Resampling.BILINEAR))
    return rgba


def run(ctx, images):
    paths = []
    try:
        mapped.to_device(ctx.model, "cuda")
        for number, source in enumerate(images):
            ctx.check_cancel()
            ctx.progress(number / len(images), f"image {number + 1}/{len(images)}")
            try:
                image = Image.open(source)
                image.load()
            except OSError:
                raise InputError("an input image is not a readable image; send a PNG, JPEG or WEBP")
            path = ctx.dir / f"image_{number}.png"
            remove_background(ctx.model, image).save(path)
            paths.append(str(path))
    finally:
        mapped.to_host(ctx.model)
        torch.cuda.empty_cache()
    return {"images": paths}
