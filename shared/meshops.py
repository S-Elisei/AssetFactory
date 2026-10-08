"""Mesh operations shared by the stages: GLB read and write, the check of a mesh input, vertex welding, welded vertex
normals, MV-Adapter's mesh normalisation scale, UV overlap measurement and QEM decimation. CPU only.

Meshes are `vertices` (N, 3) float64 and `faces` (M, 3) int64 arrays."""
import numpy as np
from context import InputError

# Candidate texels tested per chunk of uv_overlap_texels.
CHUNK_TEXELS = 2_000_000
# Largest share of the covered texel centers that may lie inside more than one UV triangle. Guessed.
OVERLAP_SHARE = 1e-3
# Largest absolute coordinate of a mesh after MV-Adapter's normalisation. Documented.
EXTENT = 0.5


def load_glb(path):
    """Returns (vertices, faces, uv) of the GLB: all triangle meshes of the scene merged in scene coordinates, vertices
    and faces exactly as stored. `uv` is (N, 2) with v pointing up, the layout write_glb takes, or None when the mesh
    has no UVs."""
    import trimesh

    mesh = trimesh.load(path, file_type="glb", force="mesh", process=False)
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), getattr(mesh.visual, "uv", None)


def load_input_mesh(path, uv_size=None):
    """Returns (vertices, faces, uv) of the GLB `path` as load_glb does. Raises InputError when the file is not a readable
    GLB or holds no triangle. With `uv_size` it also raises InputError when the mesh has no UVs or when texel centers of
    a `uv_size` x `uv_size` texture lie inside more than one UV triangle in a share above OVERLAP_SHARE of the
    covered texel centers."""
    try:
        vertices, faces, uv = load_glb(path)
    except Exception:
        raise InputError("mesh: the file is not a readable GLB; send a binary glTF (.glb) with a triangle mesh")
    if len(faces) == 0:
        raise InputError("mesh: the GLB contains no triangle mesh; send a binary glTF (.glb) with a triangle mesh")
    if uv_size is None:
        return vertices, faces, uv
    if uv is None:
        raise InputError("mesh: the GLB has no UV coordinates (TEXCOORD_0); run mesh_unwrap on it first and send its "
                         "output")
    overlap, covered = uv_overlap_texels(uv, faces, uv_size)
    if overlap > OVERLAP_SHARE * covered:
        raise InputError(f"mesh: {overlap} of the {covered} covered texels of the {uv_size} x {uv_size} texture lie "
                         "inside more than one UV triangle; run mesh_unwrap on it first and send its output")
    return vertices, faces, uv


def weld(vertices, faces):
    """Returns (vertices, faces) with the vertices of equal positions merged into one."""
    vertices, inverse = np.unique(np.asarray(vertices, np.float64) + 0.0, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1)[faces]


def welded_normals(vertices, faces):
    """Vertex normals of the mesh welded by position: vertices with equal positions (copies split at UV seams or at
    non-manifold edges) get one normal, bit-identical across the copies."""
    import trimesh

    unique, inverse = weld(vertices, np.arange(len(vertices)))
    return trimesh.Trimesh(unique, inverse[faces], process=False).vertex_normals[inverse]


def mesh_scale(vertices):
    """Returns the factor that scales `vertices` so that their largest absolute coordinate equals EXTENT."""
    return EXTENT / np.abs(vertices).max()


def decimate_mesh(mesh, target_faces):
    """Decimates the MeshLib mesh in place with QEM down to target_faces and packs it."""
    import meshlib.mrmeshpy as mr

    settings = mr.DecimateSettings()
    settings.maxError = 1e9
    settings.maxDeletedFaces = max(0, mesh.topology.numValidFaces() - target_faces)
    settings.packMesh = True
    mr.decimateMesh(mesh, settings)


def decimate(vertices, faces, target_faces):
    """Returns (vertices, faces) decimated with QEM (MeshLib, topology kept) down to target_faces; a mesh with no more
    faces is returned as is."""
    import meshlib.mrmeshnumpy as mrn

    if len(faces) <= target_faces:
        return vertices, faces
    mesh = mrn.meshFromFacesVerts(np.asarray(faces, np.int32), np.asarray(vertices, np.float32))
    decimate_mesh(mesh, target_faces)
    return np.asarray(mrn.getNumpyVerts(mesh), np.float64), np.asarray(mrn.getNumpyFaces(mesh.topology), np.int64)


def uv_overlap_texels(uv, faces, size):
    """(overlapping, covered): the numbers of texel centers of a size x size texture that lie strictly inside two or
    more UV triangles, and inside at least one. Texel centers on a shared edge, and triangles of zero area, do not
    count."""
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
    counts = np.bincount(np.concatenate(hit), minlength=1)
    return int((counts >= 2).sum()), int((counts >= 1).sum())


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
