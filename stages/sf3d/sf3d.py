"""Stage sf3d: a UV-unwrapped GLB with a base-color texture and a normal map of one object from an RGBA image with
stable-fast-3d. The object is recentered and scaled to `foreground_ratio`. The mesh is rotated by `input_elevation_deg`
about the X axis, then by 180 degrees about the Y axis. The mesh is not cleaned."""
import mapped
import numpy as np
import torch
import trimesh
from accelerate import init_empty_weights
from context import MODELS
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from sf3d.system import SF3D
from sf3d.utils import resize_foreground

REPO = "stabilityai/stable-fast-3d"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 4.0
# Parts of the model stored and kept in bf16; the other parts stay in fp32. Guessed.
BF16_PARTS = ("image_tokenizer", "backbone", "image_estimator")
# Modules, parameters and buffers, named by their path in the model, that run() never uses: the illumination estimator
# and the text tower of the CLIP model of the image estimator. They are removed after building; their tensors are not
# converted.
UNUSED = ("global_estimator", "image_estimator.model.transformer", "image_estimator.model.token_embedding",
          "image_estimator.model.ln_final", "image_estimator.model.positional_embedding",
          "image_estimator.model.text_projection", "image_estimator.model.logit_scale",
          "image_estimator.model.attn_mask")
# File that download() writes and load() reads: model.safetensors without the UNUSED tensors and with the tensors of
# BF16_PARTS in bf16.
BF16_WEIGHTS = MODELS / "sf3d" / "model.safetensors"
DEVICE = torch.device("cuda")


def download():
    """Fetches the checkpoint and the config of the image tokenizer's DINOv2, and writes BF16_WEIGHTS."""
    config = OmegaConf.load(hf_hub_download(REPO, "config.yaml"))
    hf_hub_download(config.image_tokenizer.pretrained_model_name_or_path, "config.json")
    source = hf_hub_download(REPO, "model.safetensors")
    if not BF16_WEIGHTS.exists():
        BF16_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(source, "pt") as file:
            tensors = {key: file.get_tensor(key).to(torch.bfloat16) if key.split(".")[0] in BF16_PARTS
                       else file.get_tensor(key) for key in file.keys() if not key.startswith(UNUSED)}
        mapped.save_weights(BF16_WEIGHTS, tensors)


def load():
    """Returns the SF3D model on the GPU, built without weights, without the UNUSED entries, and loaded from
    BF16_WEIGHTS with strict=False."""
    config = OmegaConf.load(hf_hub_download(REPO, "config.yaml"))
    OmegaConf.resolve(config)
    with init_empty_weights():
        model = SF3D(config)
    for path in UNUSED:
        parent, _, name = path.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, name, None)
    with safe_open(BF16_WEIGHTS, "pt", device="cuda") as file:
        model.load_state_dict({key: file.get_tensor(key) for key in file.keys()}, strict=False, assign=True)
    for part in BF16_PARTS:
        getattr(model, part).to(torch.bfloat16)
    return model.to(DEVICE).eval()


def run(ctx, image, texture_resolution, remesh, target_vertex_count, foreground_ratio, input_elevation_deg):
    ctx.progress(0.0, "preprocessing")
    foreground = resize_foreground(Image.open(image), foreground_ratio)
    ctx.check_cancel()

    ctx.progress(0.15, "reconstructing")
    np.random.seed(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mesh, _ = ctx.model.run_image(foreground, bake_resolution=texture_resolution, remesh=remesh,
                                      vertex_count=target_vertex_count)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(input_elevation_deg), [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    ctx.check_cancel()

    ctx.progress(0.9, "writing")
    path = ctx.dir / "mesh.glb"
    mesh.export(path, include_normals=True)
    return {"mesh": str(path), "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
