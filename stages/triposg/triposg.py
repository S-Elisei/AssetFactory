"""Stage triposg: the raw mesh of one object from an RGBA image with TripoSG (a rectified-flow transformer over an SDF
latent; the VAE's flash decoder evaluates the SDF and DiffDMC extracts the surface). Writes the mesh as `raw.glb`.
During surface extraction the transformer and the image encoder wait in host memory and return to the GPU at the start
of the next run."""
import shutil
from pathlib import Path

import cv2
import hub
import mapped
import numpy as np
import torch
import trimesh
from accelerate import init_empty_weights
from context import MODELS
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model
from triposg.models.autoencoders import TripoSGVAEModel
from triposg.models.transformers import TripoSGDiTModel
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from triposg.schedulers import RectifiedFlowScheduler

REPO = "VAST-AI/TripoSG"
REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 3.5
FILES = ["scheduler/*", "feature_extractor_dinov2/*", "image_encoder_dinov2/*", "transformer/*", "vae/*"]
# Weight file of each component folder of the checkpoint.
WEIGHTS = {
    "transformer": "diffusion_pytorch_model.safetensors",
    "vae": "diffusion_pytorch_model.safetensors",
    "image_encoder_dinov2": "model.safetensors",
}
# Modules of each component that run() never uses; they are deleted after building and their tensors are not converted.
UNUSED = {"vae": ("encoder", "quant")}
# Folder that download() writes and load() reads: the component folders of the checkpoint converted to fp16.
FP16 = MODELS / "triposg" / "fp16"
DEVICE = torch.device("cuda")
# Largest number of query points one VAE decode call evaluates. Guessed.
DECODE_POINTS = 20000
# Longest side in pixels of the model input before cropping. Documented.
MAX_SIDE = 2000
# Border around the cropped object, in the share of its longer side. Documented.
PADDING_RATIO = 0.1


def download():
    """Fetches the checkpoint and writes FP16; components already converted are kept."""
    snapshot = Path(hub.snapshot(REPO, REVISION, FILES))
    for name, weights in WEIGHTS.items():
        target = FP16 / name
        target.mkdir(parents=True, exist_ok=True)
        if not (target / weights).exists():
            with safe_open(snapshot / name / weights, "pt") as file:
                tensors = {key: file.get_tensor(key).to(torch.float16) for key in file.keys()
                           if key.split(".")[0] not in UNUSED.get(name, ())}
            mapped.save_weights(target / weights, tensors)
        shutil.copyfile(snapshot / name / "config.json", target / "config.json")


def _load(name, factory):
    """Returns the model that `factory` builds without weights, with the weights of FP16/`name` on the GPU."""
    with init_empty_weights():
        model = factory()
    for module in UNUSED.get(name, ()):
        delattr(model, module)
    model.load_state_dict(load_file(FP16 / name / WEIGHTS[name], device="cuda"), assign=True)
    return model.to(DEVICE, torch.float16).eval()


def load():
    """Returns {"pipeline": pipeline, "decode": the VAE's own decode method}."""
    snapshot = hub.snapshot(REPO, REVISION, FILES)
    pipeline = TripoSGPipeline(
        vae=_load("vae", lambda: TripoSGVAEModel.from_config(TripoSGVAEModel.load_config(FP16 / "vae"))),
        transformer=_load("transformer", lambda: TripoSGDiTModel.from_config(
            TripoSGDiTModel.load_config(FP16 / "transformer"))),
        scheduler=RectifiedFlowScheduler.from_pretrained(snapshot, subfolder="scheduler"),
        image_encoder_dinov2=_load("image_encoder_dinov2", lambda: Dinov2Model(
            Dinov2Config.from_pretrained(FP16 / "image_encoder_dinov2"))),
        feature_extractor_dinov2=BitImageProcessor.from_pretrained(snapshot, subfolder="feature_extractor_dinov2"),
    )
    pipeline.set_progress_bar_config(disable=True)
    return {"pipeline": pipeline, "decode": pipeline.vae.decode}


def _condition(path):
    """Returns the model input of the RGBA image at `path`: the image scaled down to MAX_SIDE, the bounding box of its
    pixels with alpha above 0 composited on white and padded to a square with a PADDING_RATIO border."""
    rgba = np.asarray(Image.open(path))
    height, width = rgba.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1:
        rgba = cv2.resize(rgba, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    rows, columns = np.nonzero(rgba[..., 3])
    top, left = rows.min(), columns.min()
    box_height, box_width = rows.max() - top + 1, columns.max() - left + 1
    alpha = rgba[..., 3:].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) / 255 * alpha + (1 - alpha)
    if box_width > box_height:
        side = int(box_width * PADDING_RATIO)
        vertical = int(side + (box_width - box_height) / 2)
    else:
        vertical = int(box_height * PADDING_RATIO)
        side = int(vertical + (box_height - box_width) / 2)
    box = rgb[top:top + box_height, left:left + box_width]
    padded = np.pad(box, ((vertical, vertical), (side, side), (0, 0)), constant_values=1.0)
    return Image.fromarray((padded * 255).astype(np.uint8))


def _chunked_decode(ctx, decode):
    """Returns `decode` for the VAE that splits the decode of one latent at more than DECODE_POINTS query points into
    calls of at most DECODE_POINTS points and raises Cancelled before each decode when the run was cancelled."""

    def chunked(z, sampled_points, **kwargs):
        ctx.check_cancel()
        if sampled_points.shape[0] > 1 or sampled_points.shape[1] <= DECODE_POINTS:
            return decode(z, sampled_points, **kwargs)
        output = decode(z, sampled_points[:, :DECODE_POINTS], **kwargs)
        rest = sampled_points[:, DECODE_POINTS:].split(DECODE_POINTS, dim=1)
        output.sample = torch.cat([output.sample] + [decode(z, points, **kwargs).sample for points in rest], 1)
        return output

    return chunked


def run(ctx, image, steps, guidance_scale, octree_depth, seed):
    pipeline = ctx.model["pipeline"]

    ctx.progress(0.0, "preprocessing")
    condition = _condition(image)
    ctx.check_cancel()

    pipeline.transformer.to(DEVICE)
    pipeline.image_encoder_dinov2.to(DEVICE)
    pipeline.vae.decode = _chunked_decode(ctx, ctx.model["decode"])

    def on_step(pipe, step, _timestep, _tensors):
        ctx.check_cancel()
        ctx.progress(0.05 + 0.55 * (step + 1) / steps, f"sampling step {step + 1}/{steps}")
        if step + 1 == steps:
            ctx.progress(0.6, "extracting surface")
            pipe.transformer.to("cpu")
            pipe.image_encoder_dinov2.to("cpu")
            torch.cuda.empty_cache()
        return {}

    vertices, faces = pipeline(
        image=condition,
        generator=torch.Generator(DEVICE).manual_seed(seed),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        flash_octree_depth=octree_depth,
        callback_on_step_end=on_step,
    ).samples[0]
    torch.cuda.empty_cache()

    ctx.progress(0.95, "writing")
    path = ctx.dir / "raw.glb"
    trimesh.Trimesh(vertices, faces, process=False).export(path)
    return {"mesh": str(path), "vertices": len(vertices), "faces": len(faces)}
