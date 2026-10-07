"""Stage mvadapter: six views of a UV-mapped mesh from one RGBA reference image with SDXL and the MV-Adapter
image-and-geometry-to-multiview adapter. The views are generated at VIEW_SIZE pixels from normal and position renders of
the mesh, upscaled with RealESRGAN x2plus and written as `view_<n>.png` (RGBA: the color is the generated view,
the alpha is the silhouette of the mesh in that view), together with `cameras.json`, the cameras of the views in the
format described in the `bake` stage. The reference image is the view from +Z (Y up). The input mesh's vertices, faces
and UVs are only read.

SDXL and the adapter run in fp16 with the UNet stored in float8. The prompt and the negative prompt are constants: their
embeddings are computed by `download` and stored; the text encoders are not loaded. The weights of the UNet, the VAE
and the condition encoder stay in memory maps of safetensors files, and `run` copies them to the GPU for the time the
pipeline runs; the RealESRGAN model is read by spandrel from its .pth file into RAM and sits on the GPU only while the
views are upscaled. `load` and `download` do not use the Worker context."""
import json
import urllib.request
from pathlib import Path

import mapped
import meshops
import mvadapter_common as common
import numpy as np
import torch
from accelerate import init_empty_weights
from context import MODELS
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download, snapshot_download
from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, get_orthogonal_camera, render
from PIL import Image
from safetensors.torch import load_file
from scripts.inference_ig2mv_sdxl import preprocess_image
from spandrel import ModelLoader
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 4.5
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILES = ["scheduler/*", "tokenizer/*", "tokenizer_2/*", "text_encoder/config.json",
              "text_encoder/model.fp16.safetensors", "text_encoder_2/config.json",
              "text_encoder_2/model.fp16.safetensors", "unet/config.json",
              "unet/diffusion_pytorch_model.fp16.safetensors"]
TEXT_WEIGHTS = "model.fp16.safetensors"
UNET_FILE = "unet/diffusion_pytorch_model.fp16.safetensors"
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
VAE_FILE = "diffusion_pytorch_model.safetensors"
VAE_FILES = ["config.json", VAE_FILE]
ADAPTER_REPO = "huanngzh/mv-adapter"
ADAPTER_FILE = "mvadapter_ig2mv_sdxl.safetensors"
UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
# Folder that download() writes and load() reads: the VAE in fp16, the UNet with the merged adapter and the condition
# encoder, the embeddings of the prompts, and the upscaler checkpoint.
FOLDER = MODELS / "mvadapter"
UPSCALER = FOLDER / "RealESRGAN_x2plus.pth"
VAE_WEIGHTS = FOLDER / "vae.safetensors"
UNET_WEIGHTS = FOLDER / "unet.safetensors"
ENCODER_WEIGHTS = FOLDER / "cond_encoder.safetensors"
EMBEDDINGS = FOLDER / "prompt_embeds.safetensors"
DEVICE = common.DEVICE
# Storage dtype of the UNet layers that layerwise casting converts; they compute in fp16.
STORAGE = torch.float8_e4m3fn

NUM_VIEWS = 6
# Side in pixels of the generated views and of the control renders. Documented.
VIEW_SIZE = 768
# Elevation and azimuth in degrees of the views: front, right, back, left, top, bottom. Documented.
ELEVATIONS = [0, 0, 0, 0, 89.99, -89.99]
AZIMUTHS = [x - 90 for x in [0, 90, 180, 270, 180, 180]]
# Distance of the orthographic cameras from the origin and half side of their image plane, in MV-Adapter mesh units.
# Documented.
DISTANCE = 1.8
HALF_WIDTH = 0.55
# Rotation that takes the input mesh's vertices and normals into MV-Adapter's frame: (x, y, z) -> (x, -z, y), so the
# input's +Y becomes MV-Adapter's up (+Z) and the input's +Z, the direction of the reference view, becomes -Y.
# Documented.
ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# Prompt and negative prompt of the views. Documented.
PROMPT = "high quality"
NEGATIVE_PROMPT = "watermark, ugly, deformed, noisy, blurry, low contrast"
# Interpolation and scale of the noise schedule shift of the adapter's scheduler. Documented.
SHIFT_MODE, SHIFT_SCALE = "interpolated", 8.0


def _pipeline(sdxl, vae):
    """Returns the MV-Adapter SDXL pipeline without text encoders of the checkpoint folder `sdxl` and the VAE folder
    `vae`, every model built without weights, with the adapter's attention processors and condition encoder in place."""
    with init_empty_weights():
        pipeline = MVAdapterI2MVSDXLPipeline(
            vae=AutoencoderKL.from_config(AutoencoderKL.load_config(vae)),
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            unet=UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(sdxl / "unet")),
            scheduler=ShiftSNRScheduler.from_scheduler(
                EulerDiscreteScheduler.from_pretrained(sdxl, subfolder="scheduler"), shift_mode=SHIFT_MODE,
                shift_scale=SHIFT_SCALE),
        )
        pipeline.init_custom_adapter(num_views=NUM_VIEWS, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0,
                                     copy_attn_weights=False)
    return pipeline


def _prompt_embeddings(sdxl):
    """Returns {"prompt", "negative", "pooled", "negative_pooled"}: the fp16 CPU embeddings of PROMPT and
    NEGATIVE_PROMPT from the text encoders of the checkpoint folder `sdxl`, computed on the GPU."""
    with init_empty_weights():
        pipeline = StableDiffusionXLPipeline(
            vae=None,
            text_encoder=CLIPTextModel(CLIPTextConfig.from_pretrained(sdxl / "text_encoder")),
            text_encoder_2=CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(sdxl / "text_encoder_2")),
            tokenizer=CLIPTokenizer.from_pretrained(sdxl, subfolder="tokenizer"),
            tokenizer_2=CLIPTokenizer.from_pretrained(sdxl, subfolder="tokenizer_2"),
            unet=None,
            scheduler=None,
        )
    for name in ("text_encoder", "text_encoder_2"):
        getattr(pipeline, name).load_state_dict(load_file(sdxl / name / TEXT_WEIGHTS, device="cuda"), assign=True)
        getattr(pipeline, name).to(DEVICE)
    with torch.no_grad():
        embeddings = pipeline.encode_prompt(PROMPT, device=DEVICE, num_images_per_prompt=1,
                                            do_classifier_free_guidance=True, negative_prompt=NEGATIVE_PROMPT)
    return dict(zip(("prompt", "negative", "pooled", "negative_pooled"), (tensor.cpu() for tensor in embeddings)))


def download():
    """Fetches the checkpoints and writes the files `load` reads; files already written are kept. The UNet file holds
    the SDXL UNet with the adapter merged in, the layers that layerwise casting converts stored in float8."""
    sdxl = Path(snapshot_download(SDXL_REPO, allow_patterns=SDXL_FILES))
    vae = Path(snapshot_download(VAE_REPO, allow_patterns=VAE_FILES))
    adapter = hf_hub_download(ADAPTER_REPO, ADAPTER_FILE)
    FOLDER.mkdir(parents=True, exist_ok=True)
    if not UPSCALER.exists():
        partial = UPSCALER.with_name(UPSCALER.name + ".part")
        urllib.request.urlretrieve(UPSCALER_URL, partial)
        partial.replace(UPSCALER)
    if not VAE_WEIGHTS.exists():
        mapped.save_weights(VAE_WEIGHTS, {key: tensor.to(torch.float16)
                                          for key, tensor in load_file(vae / VAE_FILE).items()})
    if not EMBEDDINGS.exists():
        mapped.save_weights(EMBEDDINGS, _prompt_embeddings(sdxl))
    if not (UNET_WEIGHTS.exists() and ENCODER_WEIGHTS.exists()):
        pipeline = _pipeline(sdxl, vae)
        merged = {key: tensor.to(torch.float16) for key, tensor in load_file(adapter).items()}
        unet_state = load_file(sdxl / UNET_FILE)
        unet_state.update({key: tensor for key, tensor in merged.items() if not key.startswith("adapter.")})
        pipeline.unet.load_state_dict(unet_state, assign=True)
        pipeline.cond_encoder.load_state_dict({key: tensor for key, tensor in merged.items()
                                               if key.startswith("adapter.")}, assign=True)
        pipeline.unet.enable_layerwise_casting(storage_dtype=STORAGE, compute_dtype=torch.float16)
        mapped.save_weights(ENCODER_WEIGHTS, pipeline.cond_encoder.state_dict())
        mapped.save_weights(UNET_WEIGHTS, pipeline.unet.state_dict())


def load():
    """Returns {"pipeline": the MV-Adapter SDXL pipeline without text encoders, "embeddings": the prompt embeddings
    (memory maps), "upscaler": the RealESRGAN model in fp16 on the CPU}. The parameters of the pipeline's models point
    to memory maps of their weights; the UNet's layers are stored in float8 and compute in fp16."""
    pipeline = _pipeline(Path(snapshot_download(SDXL_REPO, allow_patterns=SDXL_FILES)),
                         Path(snapshot_download(VAE_REPO, allow_patterns=VAE_FILES)))
    for model, weights in ((pipeline.vae, VAE_WEIGHTS), (pipeline.unet, UNET_WEIGHTS),
                           (pipeline.cond_encoder, ENCODER_WEIGHTS)):
        mapped.attach(model, mapped.map_tensors([weights]))
    pipeline.unet.enable_layerwise_casting(storage_dtype=STORAGE, compute_dtype=torch.float16)
    pipeline.vae.enable_slicing()
    pipeline.set_progress_bar_config(disable=True)
    return {"pipeline": pipeline, "embeddings": mapped.map_tensors([EMBEDDINGS]),
            "upscaler": ModelLoader().load_from_file(UPSCALER).eval().half()}


def _on_step(ctx, steps):
    """Returns a `callback_on_step_end` for the pipeline: it raises Cancelled when the run was cancelled and reports the
    finished step and, after the last step, the start of decoding."""

    def callback(_pipeline, step, _timestep, _tensors):
        ctx.check_cancel()
        ctx.progress(0.1 + 0.7 * (step + 1) / steps, f"sampling step {step + 1}/{steps}")
        if step + 1 == steps:
            ctx.progress(0.8, "decoding")
        return {}

    return callback


def _write_cameras(path, cameras, scale):
    """Writes `cameras` (MV-Adapter cameras of the mesh scaled by `scale` and rotated by ROTATION) as the cameras file
    of the unscaled, unrotated input mesh."""
    to_scaled = np.eye(4)
    to_scaled[:3, :3] = ROTATION * scale
    to_input_units = np.diag([1 / scale, 1 / scale, 1 / scale, 1.0])
    c2w = np.linalg.inv(to_input_units @ cameras.w2c.double().cpu().numpy() @ to_scaled)
    half = HALF_WIDTH / scale
    entries = [{"c2w": matrix.tolist(), "left": -half, "right": half, "bottom": -half, "top": half} for matrix in c2w]
    path.write_text(json.dumps({"cameras": entries}), encoding="utf-8")


def run(ctx, mesh, image, steps, guidance_scale, texture_resolution, seed):
    vertices, faces, uv = meshops.load_input_mesh(mesh, texture_resolution)
    pipeline, upscaler = ctx.model["pipeline"], ctx.model["upscaler"]

    ctx.progress(0.0, "rendering normals and positions")
    scale = common.mesh_scale(vertices)
    scaled = common.textured_mesh((vertices @ ROTATION.T * scale).astype(np.float32), faces,
                                  (meshops.welded_normals(vertices, faces) @ ROTATION.T).astype(np.float32))
    cameras = get_orthogonal_camera(
        elevation_deg=ELEVATIONS, distance=[DISTANCE] * NUM_VIEWS, left=-HALF_WIDTH, right=HALF_WIDTH,
        bottom=-HALF_WIDTH, top=HALF_WIDTH, azimuth_deg=AZIMUTHS, device=DEVICE)
    rendered = render(NVDiffRastContextWrapper(device=DEVICE), scaled, cameras, height=VIEW_SIZE, width=VIEW_SIZE,
                      render_attr=False, render_depth=False, normal_background=0.0)
    control = torch.cat([(rendered.pos + 0.5).clamp(0, 1), (rendered.normal / 2 + 0.5).clamp(0, 1)],
                        dim=-1).permute(0, 3, 1, 2)
    silhouettes = rendered.mask.cpu().numpy()
    del scaled, rendered
    reference = preprocess_image(Image.open(image), VIEW_SIZE, VIEW_SIZE)
    ctx.check_cancel()

    ctx.progress(0.1, "encoding the reference image")
    embeddings = ctx.model["embeddings"]
    with mapped.on_gpu(DEVICE, pipeline.unet, pipeline.cond_encoder, pipeline.vae):
        views = pipeline(
            prompt_embeds=embeddings["prompt"], negative_prompt_embeds=embeddings["negative"],
            pooled_prompt_embeds=embeddings["pooled"], negative_pooled_prompt_embeds=embeddings["negative_pooled"],
            height=VIEW_SIZE, width=VIEW_SIZE, num_inference_steps=steps, guidance_scale=guidance_scale,
            num_images_per_prompt=NUM_VIEWS, control_image=control, reference_image=reference,
            generator=torch.Generator(DEVICE).manual_seed(seed), callback_on_step_end=_on_step(ctx, steps)).images
    del control

    paths = []
    upscaler.to(DEVICE)
    try:
        for number, view in enumerate(views):
            ctx.check_cancel()
            ctx.progress(0.82 + 0.15 * number / NUM_VIEWS, f"upscaling view {number + 1}/{NUM_VIEWS}")
            low = torch.from_numpy(np.asarray(view.convert("RGB"))).to(DEVICE).float().div(255)
            with torch.no_grad():
                high = upscaler(low.permute(2, 0, 1)[None].half()).float().clamp(0, 1)[0].permute(1, 2, 0)
            rgba = Image.fromarray((high * 255).round().byte().cpu().numpy())
            rgba.putalpha(Image.fromarray(silhouettes[number].astype(np.uint8) * 255).resize(
                rgba.size, Image.Resampling.BILINEAR))
            paths.append(ctx.dir / f"view_{number}.png")
            rgba.save(paths[-1])
    finally:
        upscaler.to("cpu")
        torch.cuda.empty_cache()

    ctx.progress(0.97, "writing")
    _write_cameras(ctx.dir / "cameras.json", cameras, scale)
    return {"views": [str(path) for path in paths], "cameras": str(ctx.dir / "cameras.json")}
