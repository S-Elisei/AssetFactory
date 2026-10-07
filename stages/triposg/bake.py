"""Stage bake: projects the images of orthographic views of a mesh into the mesh's own UV layout with MV-Adapter's
camera projection (nvdiffrast): each texel takes the color of the views that see it, weighted by the cosine between the
surface normal and the view direction raised to PROJECTION_ALPHA, and views that see it at a grazing angle or across a
depth edge do not count. Writes the projected color atlas `atlas.png`, `covered.png` (texels inside a UV triangle) and
`valid.png` (texels at least one view filled; 255 where true) as texture images: row 0 is the top of the texture
(v = 1) and the color of a texel outside `valid.png` is black. The mesh is scaled so that its largest absolute
coordinate is 0.5 and its vertex normals are welded across UV seams. The mesh must have non-overlapping UVs.

The views are the image files `views`, in the order of the entries of the cameras file; their alpha channel is not read.
All views have the same size. The cameras file is JSON of the form

    {"cameras": [{"c2w": [[...] x 4], "left": float, "right": float, "bottom": float, "top": float}, ...]}

with one entry per view. Every camera is orthographic. All lengths are in the units of the input mesh and all matrices
are row-major. `c2w` is the rigid camera-to-world matrix in the frame of the input mesh; the camera looks along its
-z axis, its +x axis points to the right of the image and its +y axis to the top. The image plane covers x from `left`
to `right` and y from `bottom` to `top` of the camera frame, column 0 at `left` and row 0 at `top`.

`run` does not use a Worker model."""
import json
from pathlib import Path

import meshops
import mvadapter_common as common
import numpy as np
import torch
from mvadapter.utils.mesh_utils import CameraProjection, get_orthogonal_projection_matrix
from mvadapter.utils.mesh_utils.camera import Camera
from mvadapter.utils.mesh_utils.uv import uv_precompute
from PIL import Image

DEVICE = common.DEVICE
# Backend of the projection's Poisson solver, which this stage does not use.
POISSON_BACKEND = "torch-native"
# Exponent of the cosine in the blend weight of a view, minimum cosine of a view's angle to a texel's normal, depth
# gradient (per pixel, in the units of the scaled mesh) from which a texel is rejected, and the size in pixels of the
# max filter that spreads the rejection. Documented.
PROJECTION_ALPHA = 3
MIN_COSINE = 0.2
DEPTH_GRADIENT = 0.1
DEPTH_GRADIENT_DILATION = 5


def _cameras(entries, vertices, scale):
    """Returns the MV-Adapter cameras for the cameras-file `entries` and the input mesh `vertices` (the camera frame is
    that of the mesh scaled by `scale`). Each camera's near and far planes lie one bounding-box diagonal of the mesh
    before and behind the center of its bounding box."""
    c2w = np.array([entry["c2w"] for entry in entries], np.float64)
    w2c = np.linalg.inv(c2w)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    depth = -(w2c[:, 2, :3] @ ((low + high) / 2) + w2c[:, 2, 3])
    diagonal = np.linalg.norm(high - low)
    near, far = depth - diagonal, depth + diagonal
    projection = torch.cat([
        get_orthogonal_projection_matrix(1, entry["left"], entry["right"], entry["bottom"], entry["top"], n, f,
                                         device=DEVICE) for entry, n, f in zip(entries, near, far)])
    scaling = np.diag([scale, scale, scale, 1.0])
    unscaling = np.linalg.inv(scaling)
    to_tensor = lambda array: torch.tensor(array, dtype=torch.float32, device=DEVICE)
    scaled_w2c = to_tensor(scaling @ w2c @ unscaling)
    scaled_c2w = torch.linalg.inv(scaled_w2c)
    mvp = projection @ to_tensor(w2c @ unscaling)
    return Camera(c2w=scaled_c2w, w2c=scaled_w2c, proj_mtx=mvp @ scaled_c2w, mvp_mtx=mvp,
                  cam_pos=scaled_c2w[:, :3, 3])


def _mask_image(mask):
    return Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255)


def run(ctx, mesh, views, cameras, texture_size):
    ctx.progress(0.0, "preparing")
    vertices, faces, uv = meshops.load_glb(mesh)
    scale = common.mesh_scale(vertices)
    textured = common.textured_mesh((vertices * scale).astype(np.float32), faces,
                                    meshops.welded_normals(vertices, faces).astype(np.float32), uv, texture_size)
    entries = json.loads(Path(cameras).read_text(encoding="utf-8"))["cameras"]
    camera = _cameras(entries, vertices, scale)
    images = torch.from_numpy(np.stack([np.asarray(Image.open(path).convert("RGB")) for path in views])).to(DEVICE)
    ctx.check_cancel()

    ctx.progress(0.1, "projecting the views")
    projection = CameraProjection(POISSON_BACKEND, None, DEVICE)
    result = projection(
        images.float() / 255, textured, camera, uv_size=texture_size, aoi_cos_valid_threshold=MIN_COSINE,
        depth_grad_dilation=DEPTH_GRADIENT_DILATION, depth_grad_threshold=DEPTH_GRADIENT,
        uv_exp_blend_alpha=PROJECTION_ALPHA, poisson_blending=False, from_scratch=True, uv_padding=False,
        return_dict=True)
    covered = uv_precompute(projection.ctx, textured, texture_size, texture_size).uv_mask

    ctx.progress(0.9, "writing")
    paths = {name: ctx.dir / f"{name}.png" for name in ("atlas", "covered", "valid")}
    Image.fromarray((result.uv_proj.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(paths["atlas"])
    _mask_image(covered).save(paths["covered"])
    _mask_image(result.uv_proj_mask).save(paths["valid"])
    return {name: str(path) for name, path in paths.items()}
