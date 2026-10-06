"""Mesh operations shared by the stages: GLB read and write, vertex welding, welded vertex normals and UV overlap
measurement. CPU only.

Meshes are `vertices` (N, 3) float64 and `faces` (M, 3) int64 arrays."""
import numpy as np

# Candidate texels tested per chunk of uv_overlap_texels.
CHUNK_TEXELS = 2_000_000


def load_glb(path):
    """Returns (vertices, faces, uv) of the GLB: all triangle meshes of the scene merged in scene coordinates, vertices
    and faces exactly as stored. `uv` is (N, 2) with v pointing up, the layout write_glb takes, or None when the mesh
    has no UVs."""
    import trimesh

    mesh = trimesh.load(path, file_type="glb", force="mesh", process=False)
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), getattr(mesh.visual, "uv", None)


def weld(vertices, faces):
    """Returns (vertices, faces) with the vertices of equal positions merged into one."""
    vertices, inverse = np.unique(np.asarray(vertices, np.float64) + 0.0, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1)[faces]


def welded_normals(vertices, faces):
    """Vertex normals of the mesh welded by position: vertices with equal positions (copies split at UV seams or at
    non-manifold edges) get one normal, bit-identical across the copies."""
    import trimesh

    unique, inverse = np.unique(np.asarray(vertices, np.float64) + 0.0, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return trimesh.Trimesh(unique, inverse[faces], process=False).vertex_normals[inverse]


def uv_overlap_texels(uv, faces, size):
    """Number of texel centers of a size x size texture that lie strictly inside two or more UV triangles. Texel
    centers on a shared edge, and triangles of zero area, do not count."""
    tri = np.asarray(uv, np.float64)[faces] * size
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    side = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri, side = tri[side != 0], np.sign(side[side != 0])
    lo = np.maximum(np.ceil(tri.min(axis=1) - 0.5), 0).astype(np.int64)
    hi = np.minimum(np.floor(tri.max(axis=1) - 0.5), size - 1).astype(np.int64)
    nx, ny = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0), np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    total = np.cumsum(nx * ny)
    hit = []
    start = 0
    while start < len(tri):
        done = total[start - 1] if start else 0
        stop = max(int(np.searchsorted(total, done + CHUNK_TEXELS, "right")), start + 1)
        t, s, l, w, h = tri[start:stop], side[start:stop], lo[start:stop], nx[start:stop], ny[start:stop]
        owner = np.repeat(np.arange(len(t)), w * h)
        local = np.arange(len(owner)) - np.repeat(np.cumsum(w * h) - w * h, w * h)
        ix, iy = l[owner, 0] + local % w[owner], l[owner, 1] + local // w[owner]
        q = np.stack([ix + 0.5, iy + 0.5], axis=1)
        inside = np.ones(len(q), bool)
        for i in range(3):
            p, r = t[owner, i], t[owner, (i + 1) % 3]
            edge = r - p
            distance = s[owner] * (edge[:, 0] * (q[:, 1] - p[:, 1]) - edge[:, 1] * (q[:, 0] - p[:, 0]))
            inside &= distance > 1e-6 * np.linalg.norm(edge, axis=1)
        hit.append(iy[inside] * size + ix[inside])
        start = stop
    return int((np.bincount(np.concatenate(hit), minlength=1) >= 2).sum())


def write_glb(path, vertices, faces, normals, uv=None, base_color=None, normal_map=None):
    """Writes a GLB with vertex normals and a PBR material (metallicFactor 0, roughnessFactor 1). `uv` (v pointing up)
    adds TEXCOORD_0; `base_color` and `normal_map` are PIL images for the base-color and normal textures."""
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    material = PBRMaterial(baseColorTexture=base_color, normalTexture=normal_map, metallicFactor=0.0,
                           roughnessFactor=1.0)
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals, visual=TextureVisuals(uv=uv, material=material),
                           process=False)
    mesh.export(path, include_normals=True)
