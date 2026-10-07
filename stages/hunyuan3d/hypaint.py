"""Stage hypaint: textures a UV-mapped mesh from one RGBA reference image with Hunyuan3D-Paint v2.0 turbo. It writes the
base-color texture as `base_color.png` and, with a `normal_source`, the normal map as `normal_map.png`, and no mesh.
Both are size x size RGB images on the input mesh's own UV layout: image row 0 is at v = 1 of the UVs that
`meshops.load_glb` returns (v pointing up), the last row at v = 0, and columns run in increasing u. Steps in order: the
image is cropped to its object and centered on a square canvas; with `delight` it passes through the delight diffusion
model (InstructPix2Pix), otherwise it is composited on white; a multiview diffusion model generates six views of the
object from the reference image and from normal and position renders of the mesh; the views are back-projected into
the mesh's UV layout and the texels no view covers are inpainted; with a `normal_source` a tangent-space normal map is
baked from its surface onto the mesh's UV layout, on the texel grid that custom_rasterizer returns. The reference
image is the view from +Z (Y up).

The weights of every model stay in memory maps of safetensors files; `run` copies the weights of a model to the GPU for
the time that model runs. `load` and `download` do not use the Worker context."""
from functools import partial
from pathlib import Path

import cv2
import mapped
import meshops
import numpy as np
import torch
import custom_rasterizer  # must follow the import of torch
import trimesh
from accelerate import init_empty_weights
from context import MODELS
from diffusers import (AutoencoderKL, DDIMScheduler, EulerAncestralDiscreteScheduler, LCMScheduler,
                       StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel)
from huggingface_hub import snapshot_download
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.texgen.hunyuanpaint.pipeline import HunyuanPaintPipeline
from hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2p5DConditionModel
from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover
from PIL import Image
from safetensors import safe_open
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

REPO = "tencent/Hunyuan3D-2"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 4.5
DELIGHT = "hunyuan3d-delight-v2-0"
PAINT = "hunyuan3d-paint-v2-0-turbo"
# The files `load` reads from the checkpoint snapshot, and the files `download` converts.
FILES = [f"{DELIGHT}/{part}/*" for part in ("scheduler", "tokenizer", "text_encoder", "unet", "vae")] + [
    f"{PAINT}/{part}/*" for part in ("scheduler", "unet", "vae")]
# Folder that download() writes and load() reads: the weight files of the checkpoint that are not stored in fp16, each
# converted to fp16 as `<key>.safetensors`.
FP16 = MODELS / "hypaint"
TO_FP16 = {
    "delight_unet": f"{DELIGHT}/unet/diffusion_pytorch_model.safetensors",
    "paint_unet": f"{PAINT}/unet/diffusion_pytorch_model.bin",
    "paint_vae": f"{PAINT}/vae/diffusion_pytorch_model.bin",
}
DEVICE = torch.device("cuda")

# Side in pixels of the delight input, of the generated views and of the control renders. Documented.
VIEW_SIZE = 512
# Side in pixels of the renders that the views are back-projected from. Documented.
RENDER_SIZE = 2048
# (azimuth, elevation, baking weight, camera index) of the six views: front, right, back, left, top, bottom. Documented.
VIEWS = ((0, 0, 1.0, 21), (90, 0, 0.1, 12), (180, 0, 0.5, 15), (270, 0, 0.1, 18), (0, 90, 0.05, 43),
         (180, -90, 0.05, 37))
# Exponent of the view-angle cosine in the baking weight of a view. Documented.
BAKE_EXPONENT = 4
# Seed of the multiview generation. Documented.
SEED = 0
# Border around the cropped object, in the share of its width and height. Documented.
BORDER_RATIO = 0.2
# Image and text guidance scales of the delight model. Documented.
CFG_IMAGE, CFG_TEXT = 1.5, 1.0
# Offset of the start of a normal-map ray behind the surface point, in the mesh scale of the normal source. Guessed.
RAY_BACKOFF = 1e-3
# Width in texels of the band around each UV island of the normal map that is filled by inpainting, and the radius of
# the inpainting. Guessed.
BAND, INPAINT_RADIUS = 8, 3


def _from_config(cls, directory):
    return cls.from_config(cls.load_config(directory))


def _fp16_state(path):
    """Returns the tensors of the checkpoint file `path` (safetensors or pickle) in fp16."""
    if path.suffix == ".bin":
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        return {key: tensor.to(torch.float16) for key, tensor in state.items()}
    with safe_open(path, "pt") as file:
        return {key: file.get_tensor(key).to(torch.float16) for key in file.keys()}


def download():
    """Fetches the checkpoint files and writes FP16; files already converted are kept. The attention parameters of the
    paint VAE are renamed to the names its model uses."""
    snapshot = Path(snapshot_download(REPO, allow_patterns=FILES))
    FP16.mkdir(parents=True, exist_ok=True)
    for key, source in TO_FP16.items():
        target = FP16 / f"{key}.safetensors"
        if not target.exists():
            state = _fp16_state(snapshot / source)
            if key == "paint_vae":
                with init_empty_weights():
                    _from_config(AutoencoderKL, snapshot / PAINT / "vae")._fix_state_dict_keys_on_load(state)
            mapped.save_weights(target, state)


def _mapped(factory, weights):
    """Returns the model that `factory` builds without weights, its parameters and buffers pointing to memory maps of
    the safetensors file `weights`."""
    with init_empty_weights():
        model = factory()
    mapped.attach(model, mapped.map_tensors([weights]))
    return model


def load():
    """Returns {"delight": the InstructPix2Pix pipeline, "multiview": the Hunyuan3D-Paint pipeline in turbo mode}; every
    model's parameters point to memory maps of its weights, in fp16."""
    snapshot = Path(snapshot_download(REPO, allow_patterns=FILES))
    delight_directory, paint_directory = snapshot / DELIGHT, snapshot / PAINT
    text_encoder = _mapped(lambda: CLIPTextModel(CLIPTextConfig.from_pretrained(delight_directory / "text_encoder")),
                           delight_directory / "text_encoder" / "model.safetensors")
    for buffer in text_encoder.buffers():
        buffer.data = buffer.data.to(DEVICE)
    delight = StableDiffusionInstructPix2PixPipeline(
        vae=_mapped(lambda: _from_config(AutoencoderKL, delight_directory / "vae"),
                    delight_directory / "vae" / "diffusion_pytorch_model.safetensors"),
        text_encoder=text_encoder,
        tokenizer=CLIPTokenizer.from_pretrained(delight_directory / "tokenizer"),
        unet=_mapped(lambda: _from_config(UNet2DConditionModel, delight_directory / "unet"),
                     FP16 / "delight_unet.safetensors"),
        scheduler=_from_config(EulerAncestralDiscreteScheduler, delight_directory / "scheduler"),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    scheduler = DDIMScheduler.from_pretrained(paint_directory / "scheduler")
    multiview = HunyuanPaintPipeline(
        vae=_mapped(lambda: _from_config(AutoencoderKL, paint_directory / "vae"), FP16 / "paint_vae.safetensors"),
        text_encoder=None,
        tokenizer=None,
        unet=_mapped(lambda: UNet2p5DConditionModel(_from_config(UNet2DConditionModel, paint_directory / "unet")),
                     FP16 / "paint_unet.safetensors"),
        scheduler=scheduler,
        feature_extractor=None,
    )
    multiview.scheduler = LCMScheduler.from_config(scheduler.config, timestep_spacing="trailing")
    multiview.set_turbo(True)
    for pipeline in (delight, multiview):
        pipeline.set_progress_bar_config(disable=True)
    return {"delight": delight, "multiview": multiview}


def _on_gpu(pipeline):
    """Returns the context manager that holds the weights of the models of `pipeline` on the GPU for its block."""
    return mapped.on_gpu(DEVICE, *[model for model in (pipeline.text_encoder, pipeline.unet, pipeline.vae)
                                   if model is not None])


def _on_step(ctx, start, end, label):
    """Returns a `callback_on_step_end` for a diffusers pipeline: it raises Cancelled when the run was cancelled and
    reports the finished step, the steps taking the fractions `start` to `end`."""

    def callback(pipeline, step, _timestep, _tensors):
        ctx.check_cancel()
        steps = pipeline.num_timesteps
        ctx.progress(start + (end - start) * (step + 1) / steps, f"{label} step {step + 1}/{steps}")
        return {}

    return callback


class _Delight(Light_Shadow_Remover):
    """The delight step of hy3dgen around `pipeline`, a callable that runs the InstructPix2Pix pipeline."""

    def __init__(self, pipeline):
        self.device, self.cfg_image, self.cfg_text, self.pipeline = DEVICE, CFG_IMAGE, CFG_TEXT, pipeline


def _recentered(image):
    """Returns the RGBA `image` cropped to the bounding box of its pixels with alpha above 0, on a square canvas of
    transparent white, the crop centered with BORDER_RATIO of its width and height as border on every side."""
    cropped = image.crop(image.getchannel("A").getbbox())
    width, height = cropped.size
    border_x, border_y = int(width * BORDER_RATIO), int(height * BORDER_RATIO)
    side = max(width + 2 * border_x, height + 2 * border_y)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    left, top = (side - width - 2 * border_x) // 2 + border_x, (side - height - 2 * border_y) // 2 + border_y
    canvas.paste(cropped, (left, top))
    return canvas


def _white_background(image):
    """Returns the RGBA `image` scaled to VIEW_SIZE and composited on white, its alpha eroded with a 3 x 3 kernel."""
    rgba = np.array(image.resize((VIEW_SIZE, VIEW_SIZE)))
    alpha = cv2.erode(rgba[:, :, 3], np.ones((3, 3), np.uint8))[:, :, None] / 255.0
    return Image.fromarray((rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8))


def _mesh_render(vertices, faces, uv, size):
    """Returns the hy3dgen MeshRender of the mesh with `uv` (v pointing up) as its UV layout, back-projecting into a
    `size` x `size` texture."""
    render = MeshRender(default_resolution=RENDER_SIZE, texture_size=size)
    render.set_mesh(vertices, faces, vtx_uv=uv, uv_idx=faces)
    return render


def _views(ctx, pipeline, reference, normals, positions):
    """Returns the six generated views (PIL images) from the reference image and the control renders."""
    size = (VIEW_SIZE, VIEW_SIZE)
    torch.manual_seed(SEED)
    return pipeline(
        [reference],
        generator=torch.Generator().manual_seed(SEED),
        num_in_batch=len(VIEWS),
        camera_info_gen=[[view[3] for view in VIEWS]],
        camera_info_ref=[[0]],
        normal_imgs=[[image.resize(size) for image in normals]],
        position_imgs=[[image.resize(size) for image in positions]],
        callback_on_step_end=_on_step(ctx, 0.3, 0.8, "generating views"),
    ).images


def _unit(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)


def _texel_faces(faces, uv, size):
    """Rasterizes the UV layout `uv` (v pointing up) with custom_rasterizer, as clip-space positions (2u - 1, 1 - 2v),
    into a size x size image of texels. Returns (covered, face, barycentric): the (size, size) boolean mask of texels
    inside a UV triangle, the face index and the barycentric coordinates of each covered texel in row-major order."""
    position = np.stack([2 * uv[:, 0] - 1, 1 - 2 * uv[:, 1], np.zeros(len(uv)), np.ones(len(uv))], axis=1)
    index, barycentric = custom_rasterizer.rasterize(
        torch.from_numpy(position).float().to(DEVICE)[None], torch.from_numpy(faces).int().to(DEVICE), (size, size))
    index, barycentric = index.cpu().numpy(), barycentric.cpu().numpy()
    covered = index > 0
    return covered, index[covered] - 1, barycentric[covered]


def _vertex_tangents(vertices, faces, uv, normals):
    """Returns (tangents, handedness): per vertex the direction of increasing u summed over the vertex's faces and made
    perpendicular to its normal, and -1 where cross(normal, tangent) points against the direction of increasing v
    (`uv` has v pointing up), else 1. Faces of zero UV area add nothing."""
    e1, e2 = vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    d1, d2 = uv[faces[:, 1]] - uv[faces[:, 0]], uv[faces[:, 2]] - uv[faces[:, 0]]
    determinant = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=determinant != 0)[:, None]
    toward_u, toward_v = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inverse, (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inverse
    tangents, bitangents = np.zeros_like(vertices), np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(tangents, faces[:, corner], toward_u)
        np.add.at(bitangents, faces[:, corner], toward_v)
    tangents = _unit(tangents - normals * (tangents * normals).sum(axis=1, keepdims=True))
    return tangents, np.where((np.cross(normals, tangents) * bitangents).sum(axis=1) < 0, -1.0, 1.0)


def _surface_normals(vertices, faces, points, directions):
    """Returns the unit normal of the surface (`vertices`, `faces`) at each of `points`: the welded vertex normals
    interpolated at the nearer of the two first hits of the rays along `directions` and against them, each starting
    RAY_BACKOFF behind the point; a point whose two rays miss takes the normal of the nearest vertex."""
    surface = trimesh.Trimesh(vertices, faces, process=False)
    vertex_normals = meshops.welded_normals(vertices, faces)
    count = len(points)
    offset = RAY_BACKOFF * surface.scale * directions
    both = np.concatenate([points, points])
    triangle, ray, location = surface.ray.intersects_id(
        np.concatenate([points - offset, points + offset]), np.concatenate([directions, -directions]),
        multiple_hits=False, return_locations=True)
    distance = np.full(2 * count, np.inf)
    distance[ray] = np.linalg.norm(location - both[ray], axis=1)
    hit_triangle = np.full(2 * count, -1)
    hit_triangle[ray] = triangle
    hit_location = both.copy()
    hit_location[ray] = location
    pick = np.arange(count) + (distance[count:] < distance[:count]) * count
    triangle, location = hit_triangle[pick], hit_location[pick]
    hit = triangle >= 0
    normals = np.empty((count, 3))
    weights = trimesh.triangles.points_to_barycentric(surface.triangles[triangle[hit]], location[hit])
    normals[hit] = (vertex_normals[surface.faces[triangle[hit]]] * weights[:, :, None]).sum(axis=1)
    normals[~hit] = vertex_normals[surface.nearest.vertex(points[~hit])[1]]
    return _unit(normals)


def _texel_frames(vertices, faces, uv, normals, size):
    """Returns (covered, points, normal, tangent, bitangent) of the mesh surface at the texels of a size x size texture
    laid out as `uv` (v pointing up; `normals` the vertex normals), the texels of _texel_faces: the (size, size) boolean
    mask of texels inside a UV triangle, and for each covered texel in row-major order its surface point and its unit
    normal, tangent (toward increasing u) and bitangent (toward increasing v)."""
    covered, face, barycentric = _texel_faces(faces, uv, size)
    tangents, handedness = _vertex_tangents(vertices, faces, uv, normals)
    corners, weights = faces[face], barycentric[:, :, None]

    def blend(values):
        return (values[corners] * weights).sum(axis=1)

    normal = _unit(blend(normals))
    tangent = blend(tangents)
    tangent = _unit(tangent - normal * (tangent * normal).sum(axis=1, keepdims=True))
    bitangent = np.cross(normal, tangent) * np.where(blend(handedness[:, None]) < 0, -1.0, 1.0)
    return covered, blend(vertices), normal, tangent, bitangent


def _normal_map(vertices, faces, uv, normals, source_vertices, source_faces, size):
    """Returns the tangent-space normal map (uint8, size x size x 3) of the surface (`source_vertices`, `source_faces`)
    on the mesh's UV layout: texel (i, j) holds the source normal at the mesh surface point of _texel_frames, in the
    frame of the mesh's tangent, bitangent and normal there. Texels outside the UV islands hold the unperturbed normal;
    a band of BAND texels around each island is inpainted."""
    covered, points, normal, tangent, bitangent = _texel_frames(vertices, faces, uv, normals, size)
    surface = _surface_normals(source_vertices, source_faces, points, normal)
    local = np.stack([(surface * tangent).sum(axis=1), (surface * bitangent).sum(axis=1),
                      (surface * normal).sum(axis=1)], axis=1)
    image = np.empty((size, size, 3), np.float32)
    image[...] = (0.5, 0.5, 1.0)
    image[covered] = _unit(local) * 0.5 + 0.5
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    inside = covered.astype(np.uint8)
    band = cv2.dilate(inside, np.ones((3, 3), np.uint8), iterations=BAND) & (1 - inside)
    return cv2.inpaint(image, band * 255, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def _base_color(ctx, render, views):
    """Back-projects the six `views` through `render` into one texture, weighting each view by its weight and its
    view-angle cosine, and inpaints the texels no view covers. Returns the uint8 (size, size, 3) base color."""
    ctx.progress(0.82, "baking the texture")
    textures, trusts = [], []
    for view, (azimuth, elevation, weight, _) in zip(views, VIEWS):
        texture, trust, _ = render.back_project(view.resize((RENDER_SIZE, RENDER_SIZE)), elevation, azimuth)
        textures.append(texture)
        trusts.append(weight * trust ** BAKE_EXPONENT)
    texture, covered = render.fast_bake_texture(textures, trusts)
    ctx.check_cancel()
    ctx.progress(0.88, "inpainting uncovered texels")
    return render.uv_inpaint(texture, covered.squeeze(-1).cpu().numpy().astype(np.uint8) * 255)


def run(ctx, mesh, image, normal_source, texture_resolution, delight):
    vertices, faces, uv = meshops.load_input_mesh(mesh, texture_resolution)
    uv = np.asarray(uv, np.float64)

    ctx.progress(0.0, "preparing the reference image")
    reference = _recentered(Image.open(image))
    if delight:
        with _on_gpu(ctx.model["delight"]):
            remover = _Delight(partial(ctx.model["delight"],
                                       callback_on_step_end=_on_step(ctx, 0.02, 0.25, "delighting")))
            reference = remover(reference)
    else:
        reference = _white_background(reference)
    ctx.check_cancel()

    ctx.progress(0.27, "rendering normals and positions")
    render = _mesh_render(vertices, faces, uv, texture_resolution)
    normals = [render.render_normal(elevation, azimuth, use_abs_coor=True, return_type="pl")
               for azimuth, elevation, _, _ in VIEWS]
    positions = [render.render_position(elevation, azimuth, return_type="pl") for azimuth, elevation, _, _ in VIEWS]
    ctx.check_cancel()
    with _on_gpu(ctx.model["multiview"]):
        views = _views(ctx, ctx.model["multiview"], reference, normals, positions)
    ctx.check_cancel()

    base_color = _base_color(ctx, render, views)
    del render
    ctx.check_cancel()

    normal_map_path = None
    if normal_source is not None:
        ctx.progress(0.92, "baking the normal map")
        source_vertices, source_faces, _ = meshops.load_glb(normal_source)
        normal_map = _normal_map(vertices, faces, uv, meshops.welded_normals(vertices, faces), source_vertices,
                                 source_faces, texture_resolution)
        normal_map_path = ctx.dir / "normal_map.png"
        Image.fromarray(normal_map).save(normal_map_path)
        ctx.check_cancel()

    torch.cuda.empty_cache()
    base_color_path = ctx.dir / "base_color.png"
    Image.fromarray(base_color).save(base_color_path)
    return {"base_color": str(base_color_path), "normal_map": None if normal_map_path is None else str(normal_map_path)}
