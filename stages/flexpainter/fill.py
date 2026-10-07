"""Stage fill: completes the texels that the projections of `bake` left empty with FlexPainter's outpainter (TEXGen
finetuned for UV texture completion at SIZE x SIZE) and writes one completed atlas PNG per projection.

`run(ctx, mesh, covered, valid, sets)`: `mesh` is the GLB with the UVs of the projections, `covered` and `valid` the
files of the same names that `bake` returns, `sets` a list of {"atlas": path, "views": [paths]}: one entry of the
`atlases` of `bake` and the view images of that set, RGBA files whose alpha is the silhouette of the mesh (the color is
read on black where the alpha is 0). The atlas size is a multiple of SIZE. Returns {"atlases": [path, ...]}: the
completed RGB atlas of each set, in order, the same size and orientation as the atlas of `bake`.

For each set, in order: the projection is reduced to SIZE (a texel is known when at least KNOWN_SHARE of the texels it
covers are valid; its color is the mean of those); the CLIP embeddings of the views on black are averaged; the
outpainter samples the SIZE atlas conditioned on the known texels, the embeddings and the mesh positions; every covered
texel that the projection left empty takes the outpainter's color, bilinearly enlarged to the atlas size; the atlas
grows GROW texels past the UV layout by repeated averaging of filled neighbors. The mesh positions are computed once and
each set is sampled with the same seed.

The outpainter and the CLIP models run in evaluation mode. The weights of every model stay in memory maps of safetensors
files; `run` copies the weights of a model to the GPU for the time that model runs. `load` and `download` do not use the
Worker context."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import mapped
import meshops
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from context import MODELS
from huggingface_hub import hf_hub_download, snapshot_download
from model.clip import ClipTokenizer
from model.outpainter_net import OutpainterNet
from open_clip.model import _build_vision_tower
from PIL import Image
from pipeline.outpainter import OutpainterPipe
from spuv.mesh_utils import vertex_transform
from spuv.rasterize import NVDiffRasterizerContext
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 4.0
REPO = "StarYDY/FlexPainter"
CHECKPOINT = "outpainter/texgen_v1.ckpt"
OPEN_CLIP_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
OPEN_CLIP_FILES = ["open_clip_config.json", "open_clip_model.safetensors"]
ENCODER_REPO = "lambdalabs/sd-image-variations-diffusers"
ENCODER_FILES = ["image_encoder/config.json", "image_encoder/pytorch_model.bin",
                 "feature_extractor/preprocessor_config.json"]
# Folder that download() writes and load() reads: the EMA weights of the outpainter in fp32 and the weights of the CLIP
# image encoder in bf16, each as one safetensors file.
FOLDER = MODELS / "flexpainter"
OUTPAINTER_WEIGHTS = FOLDER / "outpainter.safetensors"
ENCODER_WEIGHTS = FOLDER / "image_encoder.safetensors"
DEVICE = torch.device("cuda")

# Configuration of the outpainter network. Documented.
OUTPAINTER = {
    "in_channels": 10,
    "out_channels": 3,
    "num_layers": [1, 1, 1, 1, 1],
    "point_block_num": [1, 1, 2, 4, 6],
    "block_out_channels": [32, 256, 1024, 1024, 2048],
    "dropout": [0.0, 0.0, 0.0, 0.1, 0.1],
    "use_uv_head": True,
    "block_type": ["uv", "point_uv", "uv_dit", "uv_dit", "uv_dit"],
    "voxel_size": [0.01, 0.02, 0.05, 0.05, 0.05],
    "window_size": [0, 256, 256, 512, 1024],
    "num_heads": [4, 4, 16, 16, 16],
    "skip_input": True,
    "skip_type": "adaptive",
    "weights": None,
}
# Side in texels of the outpainter's atlas, sampling steps, guidance scale and guidance interval, guidance rescale, seed
# and bounding-box half size of the mesh positions. Documented.
SIZE = 1024
STEPS = 30
CFG_SCALE = 3.5
GUIDANCE_INTERVAL = (0.0, 1.0)
GUIDANCE_RESCALE = 0.0
SEED = 42
MESH_SCALE = 0.5
# Share of valid texels from which a reduced texel counts as known, and the number of texels the atlas grows past the UV
# layout. Guessed.
KNOWN_SHARE = 0.5
GROW = 16


def _encoder(directory):
    """Returns the CLIP image encoder of the checkpoint folder `directory`, built without weights."""
    with init_empty_weights():
        return CLIPVisionModelWithProjection(CLIPVisionConfig.from_pretrained(directory / "image_encoder"))


def download():
    """Fetches the checkpoints and writes the files `load` reads; files already written are kept. The outpainter file
    holds the EMA weights under the network's parameter names."""
    checkpoint = hf_hub_download(REPO, CHECKPOINT)
    snapshot_download(OPEN_CLIP_REPO, allow_patterns=OPEN_CLIP_FILES)
    encoder = Path(snapshot_download(ENCODER_REPO, allow_patterns=ENCODER_FILES))
    FOLDER.mkdir(parents=True, exist_ok=True)
    if not OUTPAINTER_WEIGHTS.exists():
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
        names = [key[len("backbone."):] for key in state if key.startswith("backbone.")]
        mapped.save_weights(OUTPAINTER_WEIGHTS,
                            {name: state["backbone_ema." + name.replace(".", "")] for name in names})
    if not ENCODER_WEIGHTS.exists():
        state = torch.load(encoder / "image_encoder" / "pytorch_model.bin", map_location="cpu", mmap=True,
                           weights_only=True)
        mapped.save_weights(ENCODER_WEIGHTS,
                            {name: state[name].to(torch.bfloat16) for name in _encoder(encoder).state_dict()})


class _Clip:
    """The CLIP embeddings of FlexPainter's ClipTokenizer over the models of this stage: `process_image` (CLIP image
    encoder) and `process_pseudo_text` (OpenCLIP image tower)."""

    process_image = ClipTokenizer.process_image
    process_pseudo_text = ClipTokenizer.process_pseudo_text

    def __init__(self, image_encoder, extractor, visual, preprocess):
        self.device = DEVICE
        self.weight_dtype = torch.bfloat16
        self.clip_image_mean = torch.as_tensor(extractor.image_mean)[:, None, None].to(DEVICE, dtype=torch.bfloat16)
        self.clip_image_std = torch.as_tensor(extractor.image_std)[:, None, None].to(DEVICE, dtype=torch.bfloat16)
        self.openclip_image_mean = torch.tensor(preprocess["mean"], device=DEVICE)[:, None, None]
        self.openclip_image_std = torch.tensor(preprocess["std"], device=DEVICE)[:, None, None]
        self.openclip_image_size = visual.image_size[0]
        self.image_encoder, self.visual = image_encoder, visual
        self._models = {"image_encoder": image_encoder, "feature_extractor": extractor,
                        "openclip_model": SimpleNamespace(encode_image=visual)}

    def non_module(self, name):
        return self._models[name]


def load():
    """Returns {"outpainter": the outpainter network, "clip": the _Clip embedder}. The parameters of the outpainter,
    of the CLIP image encoder (bf16) and of the OpenCLIP image tower point to memory maps of their weights."""
    encoder_directory = Path(snapshot_download(ENCODER_REPO, allow_patterns=ENCODER_FILES))
    open_clip_directory = Path(snapshot_download(OPEN_CLIP_REPO, allow_patterns=OPEN_CLIP_FILES))
    config = json.loads((open_clip_directory / "open_clip_config.json").read_text(encoding="utf-8"))
    with init_empty_weights():
        outpainter = OutpainterNet(OUTPAINTER)
        visual = _build_vision_tower(config["model_cfg"]["embed_dim"], config["model_cfg"]["vision_cfg"])
    image_encoder = _encoder(encoder_directory)
    mapped.attach(outpainter, mapped.map_tensors([OUTPAINTER_WEIGHTS]))
    mapped.attach(image_encoder, mapped.map_tensors([ENCODER_WEIGHTS]))
    tensors = mapped.map_tensors([open_clip_directory / "open_clip_model.safetensors"])
    mapped.attach(visual, {key[len("visual."):]: tensor for key, tensor in tensors.items()
                           if key.startswith("visual.")})
    for buffer in image_encoder.buffers():
        buffer.data = buffer.data.to(DEVICE)
    extractor = CLIPImageProcessor.from_pretrained(encoder_directory, subfolder="feature_extractor")
    clip = _Clip(image_encoder.eval(), extractor, visual.eval(), config["preprocess_cfg"])
    return {"outpainter": outpainter.eval(), "clip": clip}


class _Steps:
    """The progress bar of OutpainterPipe for the sampling of one set: reports each finished sampling step, the steps
    taking the fractions `start` to `end` of the run, and raises Cancelled when the run was cancelled."""

    def __init__(self, ctx, start, end, label):
        self.ctx, self.start, self.end, self.label = ctx, start, end, label
        self.total = self.done = 0

    def update(self, count):
        self.done += count
        self.ctx.check_cancel()
        self.ctx.progress(self.start + (self.end - self.start) * self.done / self.total,
                          f"{self.label}sampling step {self.done}/{self.total}")


def _views_on_black(paths):
    """Returns the (N, 3, H, W) float tensor on the GPU of the RGBA images `paths` composited on black."""
    views = []
    for path in paths:
        rgba = torch.from_numpy(np.asarray(Image.open(path).convert("RGBA"))).to(DEVICE).float() / 255
        views.append((rgba[..., :3] * rgba[..., 3:]).permute(2, 0, 1))
    return torch.stack(views)


def _geometry(vertices, faces, uv):
    """Returns (position, mask): the (1, 3, SIZE, SIZE) mesh position map of the UV layout, with the mesh in
    FlexPainter's frame, and the (1, 1, SIZE, SIZE) float mask of the texels inside a UV triangle. Both have row 0 at
    v = 0."""
    positions = vertex_transform({"v_pos": torch.tensor(vertices, dtype=torch.float32, device=DEVICE)},
                                 mesh_scale=MESH_SCALE)["v_pos"]
    triangles = torch.tensor(faces, dtype=torch.int32, device=DEVICE)
    clip = torch.tensor(uv, dtype=torch.float32, device=DEVICE)[None] * 2.0 - 1.0
    context = NVDiffRasterizerContext("cuda", DEVICE)
    rasterized, _ = context.rasterize(
        torch.cat((clip, torch.zeros_like(clip[..., :1]), torch.ones_like(clip[..., :1])), dim=-1), triangles,
        (SIZE, SIZE))
    position, _ = context.interpolate_one(positions, rasterized, triangles)
    return position.permute(0, 3, 1, 2), (rasterized[..., 3:4] > 0).float().permute(0, 3, 1, 2)


def _known(valid, mask, factor):
    """Returns (valid_map, share, weight) of the (H, W) boolean `valid` texels of an atlas `factor` times SIZE wide:
    the (1, 1, H, W) float valid map, the (1, 1, SIZE, SIZE) share of valid texels in each reduced texel and the float
    weight of the known reduced texels; `mask` is the UV-triangle mask of the reduced size. Row 0 is at v = 0."""
    valid_map = torch.from_numpy(valid[::-1].copy()).float().to(DEVICE)[None, None]
    share = F.avg_pool2d(valid_map, factor)
    return valid_map, share, (share >= KNOWN_SHARE).float() * mask


def _reduced(colors, valid_map, share, weight, factor):
    """Returns the (1, 3, SIZE, SIZE) mean color of the valid texels, zero where `weight` is, of the atlas `colors`
    (H, W, 3 floats) reduced by `factor`; the other arguments are the results of _known."""
    color_map = torch.from_numpy(colors[::-1].copy()).to(DEVICE).permute(2, 0, 1)[None]
    return F.avg_pool2d(color_map * valid_map, factor) / share.clamp(min=1e-6) * weight


def _grown(texture, known):
    """Returns the float32 `texture` (H, W, 3) grown GROW texels past its `known` texels (H, W floats, 1 where known)
    by repeated averaging of the known neighbors."""
    for _ in range(GROW):
        smooth = cv2.blur(texture * known[..., None], (3, 3))
        weight = cv2.blur(known, (3, 3))
        grow = (known == 0) & (weight > 0)
        texture[grow] = smooth[grow] / weight[grow][:, None]
        known[grow] = 1
    return texture


def run(ctx, mesh, covered, valid, sets):
    vertices, faces, uv = meshops.load_glb(mesh)
    covered_texels = np.asarray(Image.open(covered)) > 127
    valid_texels = np.asarray(Image.open(valid)) > 127
    factor = covered_texels.shape[0] // SIZE
    atlases = [np.asarray(Image.open(entry["atlas"]).convert("RGB"), np.float32) / 255 for entry in sets]

    ctx.progress(0.0, "preparing the maps")
    position, mask = _geometry(vertices, faces, uv)
    valid_map, share, baked_weight = _known(valid_texels, mask, factor)
    pipe = OutpainterPipe.__new__(OutpainterPipe)
    pipe.device, pipe.dtype, pipe.clip = DEVICE, torch.bfloat16, ctx.model["clip"]
    conditions = []
    with mapped.on_gpu(DEVICE, pipe.clip.image_encoder, pipe.clip.visual), torch.no_grad():
        for number, (entry, colors) in enumerate(zip(sets, atlases)):
            ctx.check_cancel()
            ctx.progress(0.1 * (number + 1) / len(sets), f"encoding the views of set {number + 1}/{len(sets)}")
            baked_image = _reduced(colors, valid_map, share, baked_weight, factor)
            conditions.append(pipe.prepare_condition_info(None, None, _views_on_black(entry["views"]), baked_image,
                                                          baked_weight))

    flipped = mask.flip(2)
    enlarged_mask = F.interpolate(flipped, scale_factor=factor, mode="bilinear", align_corners=False)
    missing = covered_texels & ~valid_texels & (enlarged_mask[0, 0].cpu().numpy() > 0)
    pipe.outpainter = ctx.model["outpainter"]
    paths = []
    with mapped.on_gpu(DEVICE, pipe.outpainter):
        for number, (condition, colors) in enumerate(zip(conditions, atlases)):
            start = 0.1 + 0.85 * number / len(sets)
            pipe.pbar = _Steps(ctx, start, start + 0.85 / len(sets), f"set {number + 1}/{len(sets)}, ")
            pipe.prepare_condition_info = lambda *_: condition
            torch.manual_seed(SEED)
            sampled = pipe(None, None, None, condition["baked_image"], condition["baked_weight"], mask, position, STEPS,
                           CFG_SCALE, GUIDANCE_INTERVAL, GUIDANCE_RESCALE)
            enlarged = F.interpolate(sampled.flip(2) * flipped, scale_factor=factor, mode="bilinear",
                                     align_corners=False)
            fill = (enlarged / enlarged_mask.clamp(min=1e-6))[0].permute(1, 2, 0).cpu().numpy()
            texture = np.where(valid_texels[..., None], colors, 0.0).astype(np.float32)
            texture[missing] = fill[missing]
            texture = _grown(texture, (valid_texels | missing).astype(np.float32))
            paths.append(ctx.dir / f"atlas_{number}.png")
            Image.fromarray((texture.clip(0, 1) * 255).round().astype(np.uint8)).save(paths[-1])
    return {"atlases": [str(path) for path in paths]}
