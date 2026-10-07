"""Code shared by the Hunyuan3D-2 shape stages of the `hunyuan3d` environment: loading a Hunyuan3D-DiT turbo pipeline
with the turbo VAE onto the GPU, and generating the raw mesh. Imports torch and hy3dgen."""
import numpy as np
import torch
import trimesh
import yaml
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config
from safetensors import safe_open

VAE_REPO = "tencent/Hunyuan3D-2"
VAE_DIRECTORY = "hunyuan3d-vae-v2-0-turbo"
DEVICE = torch.device("cuda")
# Top-k mode of the FlashVDM volume decoder and algorithm of the surface extractor. Guessed.
TOPK_MODE = "merge"
MC_ALGO = "mc"


def _files(repo, directory):
    """Returns [(repo, file)] of the DiT config, DiT weights, VAE config and VAE weights."""
    return [(source, f"{folder}/{name}") for source, folder in ((repo, directory), (VAE_REPO, VAE_DIRECTORY))
            for name in ("config.yaml", "model.fp16.safetensors")]


def download(repo, directory):
    """Fetches the files that load() reads for the DiT folder `directory` of `repo`."""
    for source, name in _files(repo, directory):
        hf_hub_download(source, name)


def _state(path, prefix):
    """Returns the tensors of the safetensors file `path` whose names start with `prefix`, on the GPU, named without
    the prefix."""
    with safe_open(path, "pt", device="cuda") as file:
        return {key[len(prefix):]: file.get_tensor(key) for key in file.keys() if key.startswith(prefix)}


def _extract(grid_logits, **kwargs):
    """Surface extractor of the VAE: runs the marching cubes of MC_ALGO on the first volume of `grid_logits` without
    catching exceptions. Returns [(vertices, faces)]."""
    vertices, faces = SurfaceExtractors[MC_ALGO]().run(grid_logits[0], **kwargs)
    return [(vertices.astype(np.float32), np.ascontiguousarray(faces))]


def load(repo, directory):
    """Returns the Hunyuan3DDiTFlowMatchingPipeline of the DiT folder `directory` of `repo` with the turbo VAE (FlashVDM
    decoder, marching cubes). The DiT, its image encoder and the VAE are built without weights and loaded onto the GPU
    in fp16; the DiT file's own VAE and the VAE's encoder are not loaded."""
    dit_config, dit_weights, vae_config, vae_weights = [hf_hub_download(*file) for file in _files(repo, directory)]
    with open(dit_config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with open(vae_config, encoding="utf-8") as file:
        vae_config = yaml.safe_load(file)
    with init_empty_weights():
        model = instantiate_from_config(config["model"])
        conditioner = instantiate_from_config(config["conditioner"])
        vae = instantiate_from_config(vae_config)
    del vae.encoder, vae.pre_kl
    model.load_state_dict(_state(dit_weights, "model."), assign=True)
    conditioner.load_state_dict(_state(dit_weights, "conditioner."), assign=True)
    vae.load_state_dict(_state(vae_weights, ""), assign=True)
    vae.enable_flashvdm_decoder(enabled=True, adaptive_kv_selection=True, topk_mode=TOPK_MODE)
    vae.surface_extractor = _extract
    return Hunyuan3DDiTFlowMatchingPipeline(
        vae=vae,
        model=model,
        scheduler=instantiate_from_config(config["scheduler"]),
        conditioner=conditioner,
        image_processor=instantiate_from_config(config["image_processor"]),
        device=DEVICE,
        dtype=torch.float16,
    )


def generate(ctx, pipeline, condition, steps, guidance_scale, octree_resolution, num_chunks, seed):
    """Runs `pipeline` on `condition` (a PIL image, or {view tag: PIL image} for the multiview DiT), writes the raw
    mesh as `raw.glb` and returns {"mesh", "vertices", "faces"}. Vertices with a non-finite coordinate and the
    faces that use them are dropped."""

    def on_step(step, _timestep, _outputs):
        ctx.check_cancel()
        ctx.progress(0.5 * (step + 1) / steps, f"sampling step {step + 1}/{steps}")
        if step + 1 == steps:
            ctx.progress(0.5, "extracting surface")

    ctx.progress(0.0, "encoding image")
    hook = pipeline.vae.geo_decoder.register_forward_pre_hook(lambda *_: ctx.check_cancel())
    try:
        vertices, faces = pipeline(
            image=condition,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            generator=torch.Generator().manual_seed(seed),
            output_type="mesh",
            enable_pbar=False,
            callback=on_step,
            callback_steps=1,
        )[0]
    finally:
        hook.remove()
    torch.cuda.empty_cache()

    ctx.progress(0.95, "writing")
    faces = faces[:, ::-1]
    finite = np.isfinite(vertices).all(axis=1)
    faces = (np.cumsum(finite) - 1)[faces[finite[faces].all(axis=1)]]
    vertices = vertices[finite]
    path = ctx.dir / "raw.glb"
    trimesh.Trimesh(vertices, faces, process=False).export(path)
    return {"mesh": str(path), "vertices": len(vertices), "faces": len(faces)}
