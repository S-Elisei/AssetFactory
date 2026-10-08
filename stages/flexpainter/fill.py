"""Stage fill: completes the texels that the projections of `bake` left empty with FlexPainter's outpainter (TEXGen
finetuned for UV texture completion at SIZE x SIZE) and writes one completed atlas PNG per projection.

`run(ctx, mesh, covered, valid, sets, seed)`: `mesh` is the GLB with the UVs of the projections, `covered` and `valid` the
files of the same names that `bake` returns, `sets` a list of {"atlas": path, "views": [paths]}: one entry of the
`atlases` of `bake` and the view images of that set, RGBA files whose alpha is the silhouette of the mesh (the color is
read on black where the alpha is 0). The atlas size is a multiple of SIZE. Returns {"atlases": [path, ...]}: the
completed RGB atlas of each set, in order, the same size and orientation as the atlas of `bake`.

For each set, in order: the projection is reduced to SIZE (a texel is known when at least KNOWN_SHARE of the texels it
covers are valid; its color is the mean of those); the CLIP embeddings of the views on black are averaged; the
outpainter samples the SIZE atlas conditioned on the known texels, the embeddings and the mesh positions, and its
output is bilinearly enlarged to the atlas size. Every covered texel that the projection left empty (`missing`) takes
the outpainter's color, and every valid texel within FEATHER of a missing one mixes the projection and the outpainter's
color with the weight smoothstep(distance / FEATHER) on the projection. Distances are 3D distances between the texels'
positions on the mesh scaled so that its largest absolute coordinate is 0.5. Where the outpainter's color is used (the
missing texels and the mixed ones) it is first leveled and then mirrored. Leveling shifts its color and scales its
contrast to the mean and standard deviation of the projected valid texels around, taken with a Gaussian of sigma
LEVEL_SIGMA in 3D that counts only texels facing a similar way (`texel_means`), against those of the missing texels'
outpainter colors; the contrast factor is limited to CONTRAST_RANGE, and the leveled color is mixed with the unleveled
one by the projected texels' weight divided by CONFIDENCE_SHARE of the total weight (at most 1; zero where the missing
texels around have zero weight). Mirroring blends the color across the cuts between UV charts (8-connected sets of
covered texels). A cut texel is a texel the output covers with a texel of another chart (its partner chart) among its
CUT_NEIGHBORS nearest such texels within CUT_REACH. A used texel takes, for each of up to PARTNERS partner charts, the
nearest cut texel of its own chart with that partner chart among its CUT_CANDIDATES nearest cut texels within
MIRROR_REACH, and reads the color of the partner chart among the MIRROR_NEAREST texels nearest to its mirror image
across that cut texel; a partner chart none of them is of does not count. Its color becomes the mean of its own color
(weight 1) and those colors, each weighted 1 - smoothstep(distance to the cut texel / MIRROR_REACH). The atlas then
grows GROW texels past the UV layout by repeated averaging of filled neighbors. The texel positions, distances and
mirror partners are computed once and each set is sampled with `seed`.

The outpainter and the CLIP models run in evaluation mode. The weights of every model stay in memory maps of safetensors
files; `run` copies the weights of a model to the GPU for the time that model runs. `load` and `download` do not use the
Worker context."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import hub
import mapped
import meshops
import numpy as np
import texel_means
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from context import MODELS
from model.clip import ClipTokenizer
from model.outpainter_net import OutpainterNet
from open_clip.model import _build_vision_tower
from PIL import Image
from pipeline.outpainter import OutpainterPipe
from scipy.spatial import cKDTree
from spuv.mesh_utils import vertex_transform
from spuv.rasterize import NVDiffRasterizerContext
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 4.0
REPO = "StarYDY/FlexPainter"
REVISION = "f193f5264abdcff2630ae334fb5da3961d00a3b1"
CHECKPOINT = "outpainter/texgen_v1.ckpt"
OPEN_CLIP_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
OPEN_CLIP_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
OPEN_CLIP_FILES = ["open_clip_config.json", "open_clip_model.safetensors"]
ENCODER_REPO = "lambdalabs/sd-image-variations-diffusers"
ENCODER_REVISION = "42bc0ee1726b141d49f519a6ea02ccfbf073db2e"
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
# Side in texels of the outpainter's atlas, sampling steps, guidance scale and guidance interval, guidance rescale and
# bounding-box half size of the mesh positions. Documented.
SIZE = 1024
STEPS = 30
CFG_SCALE = 3.5
GUIDANCE_INTERVAL = (0.0, 1.0)
GUIDANCE_RESCALE = 0.0
MESH_SCALE = 0.5
# Share of valid texels from which a reduced texel counts as known, and the number of texels the atlas grows past the UV
# layout. Guessed.
KNOWN_SHARE = 0.5
GROW = 16
# Distance from the nearest missing texel up to which a valid texel mixes in the outpainter's color; sigma of the
# leveling's Gaussian;
# range of the factor that scales the contrast of the outpainter's color in leveling; share of the total weight that the
# projected texels' weight must reach for full leveling; largest distance between texels of two charts at a cut; largest
# distance from a cut texel at which a texel is mirrored; number of partner charts a texel mirrors from; number of
# nearest texels searched for the mirror image's texel in the partner chart; number of neighbors searched for the
# texels of another chart at a cut; number of nearest cut texels searched for a texel to mirror. Guessed.
FEATHER = 0.01
LEVEL_SIGMA = 0.008
CONTRAST_RANGE = (0.25, 1.0)
CONFIDENCE_SHARE = 0.1
CUT_REACH = 0.002
MIRROR_REACH = 0.005
PARTNERS = 2
MIRROR_NEAREST = 16
CUT_NEIGHBORS = 9
CUT_CANDIDATES = 32


def _encoder(directory):
    """Returns the CLIP image encoder of the checkpoint folder `directory`, built without weights."""
    with init_empty_weights():
        return CLIPVisionModelWithProjection(CLIPVisionConfig.from_pretrained(directory / "image_encoder"))


def download():
    """Fetches the checkpoints and writes the files `load` reads; files already written are kept. The outpainter file
    holds the EMA weights under the network's parameter names."""
    checkpoint = hub.file(REPO, REVISION, CHECKPOINT)
    hub.snapshot(OPEN_CLIP_REPO, OPEN_CLIP_REVISION, OPEN_CLIP_FILES)
    encoder = Path(hub.snapshot(ENCODER_REPO, ENCODER_REVISION, ENCODER_FILES))
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
    encoder_directory = Path(hub.snapshot(ENCODER_REPO, ENCODER_REVISION, ENCODER_FILES))
    open_clip_directory = Path(hub.snapshot(OPEN_CLIP_REPO, OPEN_CLIP_REVISION, OPEN_CLIP_FILES))
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


def _geometry(context, vertices, faces, uv):
    """Returns (position, mask): the (1, 3, SIZE, SIZE) mesh position map of the UV layout, with the mesh in
    FlexPainter's frame, and the (1, 1, SIZE, SIZE) float mask of the texels inside a UV triangle. Both have row 0 at
    v = 0. `context` is the NVDiffRasterizerContext."""
    positions = vertex_transform({"v_pos": torch.tensor(vertices, dtype=torch.float32, device=DEVICE)},
                                 mesh_scale=MESH_SCALE)["v_pos"]
    triangles = torch.tensor(faces, dtype=torch.int32, device=DEVICE)
    clip = torch.tensor(uv, dtype=torch.float32, device=DEVICE)[None] * 2.0 - 1.0
    rasterized, _ = context.rasterize(
        torch.cat((clip, torch.zeros_like(clip[..., :1]), torch.ones_like(clip[..., :1])), dim=-1), triangles,
        (SIZE, SIZE))
    position, _ = context.interpolate_one(positions, rasterized, triangles)
    return position.permute(0, 3, 1, 2), (rasterized[..., 3:4] > 0).float().permute(0, 3, 1, 2)


def _texels(context, vertices, faces, uv, size):
    """Returns (position, normal): the (size, size, 3) float32 position and unit normal of every texel inside a UV
    triangle, in the texel layout of the atlases of `bake` (row 0 at v = 1). The positions are those of the mesh scaled
    by meshops.mesh_scale, the normals are interpolated from meshops.welded_normals. `context` is the
    NVDiffRasterizerContext."""
    attributes = torch.tensor(np.concatenate([vertices * meshops.mesh_scale(vertices),
                                              meshops.welded_normals(vertices, faces)], axis=1),
                              dtype=torch.float32, device=DEVICE)
    triangles = torch.tensor(faces, dtype=torch.int32, device=DEVICE)
    clip = torch.tensor(np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1), dtype=torch.float32,
                        device=DEVICE)[None] * 2.0 - 1.0
    rasterized, _ = context.rasterize(
        torch.cat((clip, torch.zeros_like(clip[..., :1]), torch.ones_like(clip[..., :1])), dim=-1), triangles,
        (size, size))
    texels = context.interpolate_one(attributes, rasterized, triangles)[0][0].cpu().numpy()
    normal = texels[..., 3:]
    return texels[..., :3], normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)


def _feather(position, valid, has, missing):
    """Returns (selected, weight, used) for the (H, W) boolean `valid`, `has` (texels the outpainter's output covers)
    and `missing` texels, with `position` their (H, W, 3) positions: the mask of the valid texels the output covers, the
    float weight of the projection (the smoothstep of the distance to the nearest missing texel over FEATHER at the
    selected texels, 1 at the other valid texels, 0 elsewhere) and the boolean mask of the texels that take the
    outpainter's color at least in part."""
    selected = valid & has
    distance = cKDTree(position[missing]).query(position[selected], k=1, workers=-1)[0]
    fraction = np.clip(distance / FEATHER, 0, 1)
    weight = valid.astype(np.float32)
    weight[selected] = fraction * fraction * (3 - 2 * fraction)
    used = missing.copy()
    used[selected] = distance < FEATHER
    return selected, weight, used


def _leveled(position, normal, valid, missing, used, colors, output):
    """Levels the (H, W, 3) outpainter's `output` in place at the `used` texels against the (H, W, 3) projected `colors`
    at the `valid` texels (`missing` is the boolean mask of the texels whose output is the fill): see the module
    docstring. `position` and `normal` are the (H, W, 3) texel positions and unit normals."""
    if not used.any():
        return
    ref_weight, (ref_mean, ref_square) = texel_means.smooth_means(
        position, normal, valid, used, [colors, colors ** 2], LEVEL_SIGMA)
    fill_weight, (fill_mean, fill_square) = texel_means.smooth_means(
        position, normal, missing, used, [output, output ** 2], LEVEL_SIGMA)
    ref_std = np.sqrt(np.maximum(ref_square - ref_mean ** 2, 1e-6))
    fill_std = np.sqrt(np.maximum(fill_square - fill_mean ** 2, 1e-6))
    confidence = (np.clip(ref_weight / np.maximum(CONFIDENCE_SHARE * (ref_weight + fill_weight), 1e-6), 0, 1)
                  * (fill_weight > 1e-6))[:, None]
    leveled = ref_mean + (output[used] - fill_mean) * np.clip(ref_std / fill_std, *CONTRAST_RANGE)
    output[used] = np.clip(confidence * leveled + (1 - confidence) * output[used], 0, 1)


def _mirror_plan(position, covered, has, used):
    """Returns (texels, pairs) for mirroring across the UV chart cuts. `position` is the (H, W, 3) texel positions;
    `covered`, `has` and `used` are (H, W) boolean masks. `texels` indexes the used texels among the `has` texels
    (row-major order). `pairs` has one entry (index, source, weight) per partner rank: the indices into `texels` of the
    texels that have a partner of that rank, the index among the `has` texels of the texel each takes its mirrored color
    from, and the float32 weight of that color."""
    _, chart = cv2.connectedComponents(covered.astype(np.uint8), connectivity=8)
    points, label = position[has], chart[has]
    tree = cKDTree(points)
    distance, neighbor = tree.query(points, k=CUT_NEIGHBORS, distance_upper_bound=CUT_REACH, workers=-1)
    neighbor_label = label[np.minimum(neighbor, len(label) - 1)]
    other = np.isfinite(distance) & (neighbor_label != label[:, None])
    cut = other.any(axis=1)
    texels = np.nonzero(used[has])[0]
    if not cut.any():
        return texels, []
    partner = neighbor_label[np.arange(len(label)), other.argmax(axis=1)][cut]
    cut_points, cut_label = points[cut], label[cut]
    distance, nearest = cKDTree(cut_points).query(points[texels], k=CUT_CANDIDATES, distance_upper_bound=MIRROR_REACH,
                                                  workers=-1)
    nearest = np.minimum(nearest, len(cut_points) - 1)
    own = np.isfinite(distance) & (cut_label[nearest] == label[texels][:, None])
    taken = np.full((len(texels), PARTNERS), -1)
    pairs = []
    for rank in range(PARTNERS):
        pick = own & (partner[nearest][..., None] != taken[:, None, :]).all(axis=2)
        found = pick.any(axis=1)
        column = pick.argmax(axis=1)[found]
        index = np.nonzero(found)[0]
        cut_index, reach = nearest[index, column], distance[index, column]
        target = partner[cut_index]
        taken[index, rank] = target
        _, candidates = tree.query(2 * cut_points[cut_index] - points[texels[index]], k=MIRROR_NEAREST, workers=-1)
        hit = label[candidates] == target[:, None]
        source = candidates[np.arange(len(index)), hit.argmax(axis=1)]
        fraction = np.clip(reach / MIRROR_REACH, 0, 1)
        pairs.append((index, source, ((1 - fraction * fraction * (3 - 2 * fraction)) * hit.any(axis=1)
                                      ).astype(np.float32)))
    return texels, pairs


def _mirrored(output, has, plan):
    """Mirrors the (H, W, 3) outpainter's `output` in place across the UV chart cuts at the used texels; `has` is the
    boolean mask of the texels it covers and `plan` the result of _mirror_plan. The sources are read before any texel
    changes."""
    texels, pairs = plan
    values = output[has]
    total, weight = values[texels].copy(), np.ones(len(texels), np.float32)
    for index, source, source_weight in pairs:
        total[index] += source_weight[:, None] * values[source]
        weight[index] += source_weight
    values[texels] = total / weight[:, None]
    output[has] = values


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


def run(ctx, mesh, covered, valid, sets, seed):
    vertices, faces, uv = meshops.load_glb(mesh)
    covered_texels = np.asarray(Image.open(covered)) > 127
    valid_texels = np.asarray(Image.open(valid)) > 127
    factor = covered_texels.shape[0] // SIZE
    atlases = [np.asarray(Image.open(entry["atlas"]).convert("RGB"), np.float32) / 255 for entry in sets]

    ctx.progress(0.0, "preparing the maps")
    context = NVDiffRasterizerContext("cuda", DEVICE)
    position, mask = _geometry(context, vertices, faces, uv)
    flipped = mask.flip(2)
    enlarged_mask = F.interpolate(flipped, scale_factor=factor, mode="bilinear", align_corners=False)
    has = covered_texels & (enlarged_mask[0, 0].cpu().numpy() > 0)
    missing = has & ~valid_texels
    texel_position, texel_normal = _texels(context, vertices, faces, uv, covered_texels.shape[0])
    del context
    ctx.check_cancel()
    ctx.progress(0.01, "measuring the distances to the missing texels")
    selected, feather, used = _feather(texel_position, valid_texels, has, missing)
    ctx.check_cancel()
    ctx.progress(0.02, "finding the UV cuts")
    plan = _mirror_plan(texel_position, covered_texels, has, used)
    ctx.check_cancel()
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

    pipe.outpainter = ctx.model["outpainter"]
    outputs = []
    with mapped.on_gpu(DEVICE, pipe.outpainter):
        for number, condition in enumerate(conditions):
            start = 0.1 + 0.7 * number / len(sets)
            pipe.pbar = _Steps(ctx, start, start + 0.7 / len(sets), f"set {number + 1}/{len(sets)}, ")
            pipe.prepare_condition_info = lambda *_: condition
            torch.manual_seed(seed)
            sampled = pipe(None, None, None, condition["baked_image"], condition["baked_weight"], mask, position, STEPS,
                           CFG_SCALE, GUIDANCE_INTERVAL, GUIDANCE_RESCALE)
            enlarged = F.interpolate(sampled.flip(2) * flipped, scale_factor=factor, mode="bilinear",
                                     align_corners=False)
            outputs.append((enlarged / enlarged_mask.clamp(min=1e-6))[0].permute(1, 2, 0).cpu().numpy())

    paths = []
    for number, (colors, output) in enumerate(zip(atlases, outputs)):
        start = 0.8 + 0.2 * number / len(sets)
        ctx.progress(start, f"leveling set {number + 1}/{len(sets)}")
        _leveled(texel_position, texel_normal, valid_texels, missing, used, colors, output)
        ctx.check_cancel()
        ctx.progress(start + 0.1 / len(sets), f"mirroring set {number + 1}/{len(sets)}")
        _mirrored(output, has, plan)
        ctx.check_cancel()
        texture = np.where(valid_texels[..., None], colors, 0.0).astype(np.float32)
        texture[missing] = output[missing]
        texture[selected] = (feather[selected, None] * colors[selected]
                             + (1 - feather[selected, None]) * output[selected])
        texture = _grown(texture, (valid_texels | missing).astype(np.float32))
        paths.append(ctx.dir / f"atlas_{number}.png")
        Image.fromarray((texture.clip(0, 1) * 255).round().astype(np.uint8)).save(paths[-1])
    return {"atlases": [str(path) for path in paths]}
