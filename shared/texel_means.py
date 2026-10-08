"""Normal-aware Gaussian means over the texels of a UV layout, shared by the stages `bake` and `fill`: the mean of
per-texel values over the texels near a texel in 3D, where a texel counts the more the closer its surface normal is to
the normal of the texel the mean is taken for. Imports numpy and scipy only."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# The 14 directions of the affinity: the six axes and the eight cube diagonals, unit length.
DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
                      + [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)


def _affinity(normals):
    """Returns (N, 14): a_k(n) = clip((n . d_k - 0.5) / 0.5, 0, 1)^2 for the unit `normals` (N, 3) and the directions
    d_k."""
    return np.clip((normals @ DIRECTIONS.T - 0.5) / 0.5, 0, 1) ** 2


def smooth_means(pos, nrm, src, query, values, sigma):
    """For each texel of the boolean (H, W) `query`, returns the Gaussian (`sigma`, 3D, on a voxel grid of step
    `sigma` / 2) weighted sum of the texels of the boolean (H, W) `src` and the weighted mean of each (H, W, C) array of
    `values` over them. `pos` and `nrm` are the (H, W, 3) position and unit normal of every texel. The weight of texel t
    for texel x is G(x - t) sum_k a_k(n_x) a_k(n_t).
    Returns (weight (Q,), [mean (Q, C) per array]) with Q the number of texels of `query`, in row-major order."""
    step = sigma / 2
    covered = src | query
    low = pos[covered].min(0) - 3 * sigma
    shape = tuple(np.ceil((pos[covered].max(0) + 3 * sigma - low) / step).astype(int) + 1)
    count = int(src.sum())
    cell = np.ravel_multi_index(np.floor((pos[src] - low) / step).astype(int).T, shape)
    coords = ((pos[query] - low) / step - 0.5).T
    query_affinity, source_affinity = _affinity(nrm[query]), _affinity(nrm[src])
    columns = [array[src].reshape(count, -1) for array in values]
    channels = [np.ones((count, 1), np.float32)] + columns
    sums = np.zeros((int(query.sum()), sum(channel.shape[1] for channel in channels)), np.float32)
    size = int(np.prod(shape))
    for k in range(len(DIRECTIONS)):
        if not query_affinity[:, k].any() or not source_affinity[:, k].any():
            continue
        j = 0
        for channel in channels:
            for i in range(channel.shape[1]):
                grid = np.bincount(cell, weights=source_affinity[:, k] * channel[:, i], minlength=size
                                   ).astype(np.float32).reshape(shape)
                sums[:, j] += query_affinity[:, k] * map_coordinates(gaussian_filter(grid, sigma / step, truncate=3),
                                                                     coords, order=1)
                j += 1
    weight = sums[:, 0]
    means, j = [], 1
    for column in columns:
        means.append(sums[:, j:j + column.shape[1]] / np.maximum(weight, 1e-6)[:, None])
        j += column.shape[1]
    return weight, means
