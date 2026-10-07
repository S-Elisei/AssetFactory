"""Stage bake: projects the images of orthographic views of a mesh into the mesh's own UV layout with MV-Adapter's
camera projection (nvdiffrast): each texel takes the color of the views that see it, weighted by the cosine between the
surface normal and the view direction raised to PROJECTION_ALPHA, and views that see it at a grazing angle or across a
depth edge do not count. The mesh is scaled so that its largest absolute coordinate is 0.5 and its vertex normals are
welded across UV seams. The mesh must have non-overlapping UVs.

`view_sets` is a list of view sets, each a list of image files in the order of the entries of the cameras file; their
alpha channel is not read. All views of all sets have the same size. The sets share the mesh, the cameras file and
`texture_size`; the geometry of the projection is computed once. Returns

    {"covered": path, "valid": path, "atlases": [path, ...]}

`covered` is the PNG mask of the texels inside a UV triangle and `valid` the PNG mask of the texels that at least one
view sees validly (255 where true); both are the same for all sets. `atlases` holds the RGB PNG of the projected
colors of each set, in order. All are `texture_size` square, row 0 is the top of the texture (v = 1), and the color of a
texel outside `valid` is black.

The cameras file is JSON of the form

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
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, get_orthogonal_projection_matrix
from mvadapter.utils.mesh_utils.camera import Camera
from mvadapter.utils.mesh_utils.uv import (ExponentialBlend, SimpleUVValidityStrategy, uv_blend, uv_precompute,
                                           uv_render_attr, uv_render_geometry)
from PIL import Image

DEVICE = common.DEVICE
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


def run(ctx, mesh, view_sets, cameras, texture_size):
    ctx.progress(0.0, "preparing")
    vertices, faces, uv = meshops.load_glb(mesh)
    scale = common.mesh_scale(vertices)
    textured = common.textured_mesh((vertices * scale).astype(np.float32), faces,
                                    meshops.welded_normals(vertices, faces).astype(np.float32), uv, texture_size)
    entries = json.loads(Path(cameras).read_text(encoding="utf-8"))["cameras"]
    camera = _cameras(entries, vertices, scale)
    context = NVDiffRastContextWrapper(DEVICE)
    precomputed = uv_precompute(context, textured, texture_size, texture_size)
    width, height = Image.open(view_sets[0][0]).size
    geometry = uv_render_geometry(
        context, textured, camera, view_height=height, view_width=width, uv_precompute_output=precomputed,
        compute_depth_grad=True, depth_grad_dilation=DEPTH_GRADIENT_DILATION)
    validity = SimpleUVValidityStrategy(aoi_cos_thresh=MIN_COSINE, depth_grad_thresh=DEPTH_GRADIENT)
    covered_path, valid_path = ctx.dir / "covered.png", ctx.dir / "valid.png"
    _mask_image(precomputed.uv_mask).save(covered_path)
    _mask_image(validity(precomputed, geometry, None).any(dim=0)).save(valid_path)
    ctx.check_cancel()

    atlases = []
    for number, views in enumerate(view_sets):
        ctx.progress(0.1 + 0.9 * number / len(view_sets), f"projecting view set {number + 1}/{len(view_sets)}")
        images = torch.from_numpy(np.stack([np.asarray(Image.open(path).convert("RGB")) for path in views]))
        blended = uv_blend(
            precomputed, geometry, uv_render_attr(images.to(DEVICE).float() / 255, geometry),
            uv_validity_strategy=validity, uv_blend_weight_strategy=ExponentialBlend(alpha=PROJECTION_ALPHA),
            do_uv_padding=False, poisson_blending=False)
        atlases.append(ctx.dir / f"atlas_{number}.png")
        Image.fromarray((blended.uv_attr_blend.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(atlases[-1])
        ctx.check_cancel()
    return {"covered": str(covered_path), "valid": str(valid_path), "atlases": [str(path) for path in atlases]}
