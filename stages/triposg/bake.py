"""Stage bake: projects the images of orthographic views of a mesh into the mesh's own UV layout with MV-Adapter's
camera projection (nvdiffrast): each texel takes the color of the views that see it, weighted by the cosine between the
surface normal and the view direction raised to PROJECTION_ALPHA, and views that see it at a grazing angle or across a
depth edge do not count. A view whose camera looks along the Y axis (the meshes are Y up) needs a larger cosine than the
others. The mesh is scaled so that its largest absolute coordinate is 0.5 and its vertex normals are welded across UV
seams. The mesh must have non-overlapping UVs.

`view_sets` is a list of view sets, each a list of RGBA image files in the order of the entries of the cameras file;
the alpha channel is the silhouette of the mesh in the view. A pixel of a view is masked out when it lies outside the
silhouette or in its outline halo (`_view_mask`); a pixel masked out in a view of any set is masked out in the same
view of every set, and a view does not count at a texel that projects onto a masked-out pixel. The weight of a view at
a texel is its normalized blend weight (its cosine weight divided by the sum over the views that count at the texel)
averaged with a normal-aware Gaussian of sigma WEIGHT_SIGMA over the surface around the texel (`texel_means`: only
surfaces facing a similar way contribute), and is zero where the view does not count; the same weights blend every
set. All views of all sets have the same size. The sets share the mesh, the cameras file and `texture_size`; the
geometry of the projection is computed once. Returns

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

import cv2
import meshops
import mvadapter_common as common
import numpy as np
import texel_means
import torch
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, get_orthogonal_projection_matrix
from mvadapter.utils.mesh_utils.camera import Camera
from mvadapter.utils.mesh_utils.uv import (ExponentialBlend, SimpleUVValidityStrategy, uv_precompute, uv_render_attr,
                                           uv_render_geometry)
from PIL import Image

DEVICE = common.DEVICE
# Exponent of the cosine in the blend weight of a view, minimum cosine of a view's angle to a texel's normal, depth
# gradient (per pixel, in the units of the scaled mesh) from which a texel is rejected, and the size in pixels of the
# max filter that spreads the rejection. Documented.
PROJECTION_ALPHA = 3
MIN_COSINE = 0.2
DEPTH_GRADIENT = 0.1
DEPTH_GRADIENT_DILATION = 5
# Luma weights of the red, green and blue channels. Documented.
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
# Width in pixels of the band inside the silhouette's edge in which pixels can be halo, margin (in 0..1) by which a
# halo pixel's luma and blue-minus-red exceed those of the figure just inside the band, largest distance (RGB, 0..1)
# of a halo pixel's color to the background's, smallest projected view mask of a texel that counts for a view,
# smallest absolute Y component of the z axis of a camera that looks along the Y axis, minimum cosine of such a view,
# and sigma of the weight averaging (in the units of the scaled mesh). Guessed.
HALO_BAND = 5
HALO_MARGIN = 0.04
HALO_BACKGROUND = 0.05
MASK_MIN = 0.99
VERTICAL = 0.99
VERTICAL_MIN_COSINE = 0.5
WEIGHT_SIGMA = 0.015
# The alpha above which a view pixel is in the silhouette; the inner and outer distances (pixels past HALO_BAND) of the
# ring whose smoothed color is the figure's reference color; the sigma (pixels) of that smoothing; the distance
# (pixels) from the silhouette up to which pixels are sampled for the background color; and the growth (pixels) of the
# halo. Guessed.
SOLID_ALPHA = 0.5
RING_INNER = 1
RING_OUTER = 3
RING_SIGMA = 2
BACKGROUND_REACH = 10
HALO_GROWTH = 1


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


def _view_mask(view):
    """Returns the boolean (H, W) mask of the pixels of the float RGBA `view` (H, W, 4, values 0..1) that stay in use:
    the silhouette (alpha above SOLID_ALPHA) without its halo. The halo is the pixels of the silhouette within HALO_BAND
    of its edge whose color is lighter and bluer, by HALO_MARGIN, than the Gaussian-smoothed (RING_SIGMA) color of the
    nearest pixel of the ring at HALO_BAND + RING_INNER to HALO_BAND + RING_OUTER from the edge, or within
    HALO_BACKGROUND of the median color of the pixels outside the silhouette up to BACKGROUND_REACH pixels from it,
    grown by HALO_GROWTH pixels."""
    rgb = view[..., :3]
    solid = (view[..., 3] > SOLID_ALPHA).astype(np.uint8)
    distance = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    band = (solid > 0) & (distance <= HALO_BAND)
    ring = ((distance > HALO_BAND + RING_INNER) & (distance <= HALO_BAND + RING_OUTER)).astype(np.float32)
    smooth = (cv2.GaussianBlur(rgb * ring[..., None], (0, 0), RING_SIGMA)
              / np.maximum(cv2.GaussianBlur(ring, (0, 0), RING_SIGMA), 1e-6)[..., None])
    _, labels = cv2.distanceTransformWithLabels((ring == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    rows, columns = np.nonzero(ring)
    position = np.zeros((labels.max() + 1, 2), np.int64)
    position[labels[rows, columns]] = np.stack([rows, columns], axis=1)
    nearest = position[labels]
    reference = smooth[nearest[..., 0], nearest[..., 1]]
    lighter_bluer = ((rgb @ LUMA > reference @ LUMA + HALO_MARGIN)
                     & (rgb[..., 2] - rgb[..., 0] > reference[..., 2] - reference[..., 0] + HALO_MARGIN))
    reach = np.ones((2 * BACKGROUND_REACH + 1,) * 2, np.uint8)
    outside = (solid == 0) & (cv2.dilate(solid, reach) > 0)
    like_background = np.linalg.norm(rgb - np.median(rgb[outside], axis=0), axis=-1) < HALO_BACKGROUND
    growth = np.ones((2 * HALO_GROWTH + 1,) * 2, np.uint8)
    halo = cv2.dilate((band & (lighter_bluer | like_background)).astype(np.uint8), growth) > 0
    return (solid > 0) & ~halo


def _mask_image(mask):
    return Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255)


def run(ctx, mesh, view_sets, cameras, texture_size):
    ctx.progress(0.0, "preparing")
    vertices, faces, uv = meshops.load_glb(mesh)
    scale = meshops.mesh_scale(vertices)
    normals = meshops.welded_normals(vertices, faces).astype(np.float32)
    textured = common.textured_mesh((vertices * scale).astype(np.float32), faces, normals, uv, texture_size)
    entries = json.loads(Path(cameras).read_text(encoding="utf-8"))["cameras"]
    camera = _cameras(entries, vertices, scale)
    vertical = camera.c2w[:, 1, 2].abs() > VERTICAL
    context = NVDiffRastContextWrapper(DEVICE)
    precomputed = uv_precompute(context, textured, texture_size, texture_size)
    # The normals interpolated in the texel layout of `precomputed.uv_pos`: the same precompute of a mesh whose
    # positions are the vertex normals.
    texel_normals = uv_precompute(context, common.textured_mesh(normals, faces, normals, uv, texture_size),
                                  texture_size, texture_size).uv_pos.cpu().numpy()
    texel_normals /= np.maximum(np.linalg.norm(texel_normals, axis=-1, keepdims=True), 1e-6)
    ctx.check_cancel()

    rgba_sets, masks = [], []
    for number, views in enumerate(view_sets):
        ctx.progress(0.02 + 0.18 * number / len(view_sets), f"masking view set {number + 1}/{len(view_sets)}")
        rgba_sets.append(np.stack([np.asarray(Image.open(path).convert("RGBA")) for path in views]))
        masks.append([_view_mask(view.astype(np.float32) / 255) for view in rgba_sets[-1]])
        ctx.check_cancel()
    mask = torch.from_numpy(np.logical_and.reduce(masks)).to(DEVICE).float()

    ctx.progress(0.2, "projecting")
    height, width = rgba_sets[0].shape[1:3]
    geometry = uv_render_geometry(
        context, textured, camera, view_height=height, view_width=width, uv_precompute_output=precomputed,
        compute_depth_grad=True, depth_grad_dilation=DEPTH_GRADIENT_DILATION)
    projections = [uv_render_attr(torch.from_numpy(rgba[..., :3]).to(DEVICE).float() / 255, geometry,
                                  masks=mask if number == 0 else None) for number, rgba in enumerate(rgba_sets)]
    del mask
    validity = SimpleUVValidityStrategy(aoi_cos_thresh=MIN_COSINE, depth_grad_thresh=DEPTH_GRADIENT,
                                        mask_thresh=MASK_MIN)
    view_valid = validity(precomputed, geometry, projections[0])
    view_valid[vertical] &= geometry.uv_aoi_cos[vertical] > VERTICAL_MIN_COSINE
    covered_path, valid_path = ctx.dir / "covered.png", ctx.dir / "valid.png"
    _mask_image(precomputed.uv_mask).save(covered_path)
    _mask_image(view_valid.any(dim=0)).save(valid_path)
    ctx.check_cancel()

    ctx.progress(0.5, "averaging the view weights")
    weights = ExponentialBlend(alpha=PROJECTION_ALPHA)(precomputed, geometry, None, view_valid).cpu().numpy()
    del geometry
    covered = precomputed.uv_mask.cpu().numpy()
    _, (mean,) = texel_means.smooth_means(precomputed.uv_pos.cpu().numpy(), texel_normals, covered, covered,
                                          [np.moveaxis(weights, 0, -1)], WEIGHT_SIGMA)
    smoothed = np.zeros_like(weights)
    smoothed[:, covered] = mean.T
    weights = torch.from_numpy(smoothed * (weights > 0)).to(DEVICE)
    total = weights.sum(dim=0).clamp(min=1e-8)
    ctx.check_cancel()

    atlases = []
    for number, projection in enumerate(projections):
        ctx.progress(0.6 + 0.4 * number / len(projections), f"blending view set {number + 1}/{len(projections)}")
        blended = (weights[..., None] * projection.uv_attr_proj).sum(dim=0) / total[..., None]
        atlases.append(ctx.dir / f"atlas_{number}.png")
        Image.fromarray((blended.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(atlases[-1])
        ctx.check_cancel()
    return {"covered": str(covered_path), "valid": str(valid_path), "atlases": [str(path) for path in atlases]}
