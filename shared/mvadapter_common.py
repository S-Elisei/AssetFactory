"""Code shared by the MV-Adapter stages `mvadapter` and `bake` of the `triposg` environment: the MV-Adapter mesh object
built from arrays. Imports torch and the MV-Adapter checkout."""
import numpy as np
import torch
import torch.nn.functional as F
from mvadapter.utils.mesh_utils.mesh import TexturedMesh

DEVICE = torch.device("cuda")


def textured_mesh(vertices, faces, normals, uv=None, texture_size=None):
    """Returns the MV-Adapter mesh on the GPU of the float32 `vertices` and int64 `faces` with per-vertex `normals`
    (same indexing as `vertices`). With `uv` (v pointing up) it also holds the texture coordinates and an empty
    `texture_size` x `texture_size` texture."""
    index = torch.from_numpy(faces)
    mesh = TexturedMesh(v_pos=torch.from_numpy(vertices), t_pos_idx=index)
    if uv is not None:
        mesh.v_tex = torch.tensor(np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1), dtype=torch.float32)
        mesh.t_tex_idx = index.clone()
        mesh.texture = torch.zeros((texture_size, texture_size, 3), dtype=torch.float32)
    mesh.set_vertex_normal(F.normalize(torch.from_numpy(normals), dim=-1))
    mesh.set_stitched_mesh(mesh.v_pos, mesh.t_pos_idx)
    mesh.to(DEVICE)
    return mesh
