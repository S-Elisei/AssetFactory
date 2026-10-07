"""Stage clean: turns the mesh of a shape model into one closed manifold surface and writes it as a GLB with welded
vertex normals. Steps in order: weld the vertices of equal positions; take the level-0 surface of the generalized
winding number on a grid of GRID_STEP median edges and decimate it back to the input face count; split it into
connected pieces, the piece with the largest volume being the main body; remove the other pieces that are thinner than
THIN or farther than FAR; repeatedly move the remaining piece nearest to the assembled body along its shortest
connection until it overlaps the body by OVERLAP grid steps and unite the two."""
import igl
import manifold3d
import meshlib.mrmeshnumpy as mrn
import meshlib.mrmeshpy as mr
import meshops
import numpy as np

# Grid step of the level set, in median input edge lengths. Guessed.
GRID_STEP = 1 / 3
# Separate pieces with 2 * volume / area below this share of the model height are removed. Guessed.
THIN = 0.003
# Separate pieces farther from the main body than this share of their largest extent are removed. Guessed.
FAR = 0.5
# Attached pieces overlap the body by this many grid steps. Guessed.
OVERLAP = 1.0


def _to_manifold(vertices, faces):
    mesh = manifold3d.Mesh(vert_properties=np.asarray(vertices, np.float32), tri_verts=np.asarray(faces, np.uint32))
    return manifold3d.Manifold(mesh)


def _from_manifold(manifold):
    mesh = manifold.to_mesh()
    return np.asarray(mesh.vert_properties[:, :3], np.float64), np.asarray(mesh.tri_verts, np.int64)


def run(ctx, mesh):
    vertices, faces, _ = meshops.load_glb(mesh)

    ctx.progress(0.0, "welding")
    vertices, faces = meshops.weld(vertices, faces)
    step = GRID_STEP * float(np.median(np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1)))
    height = float(np.ptp(vertices[:, 1]))
    ctx.check_cancel()

    ctx.progress(0.05, "rebuilding the surface")
    params = mr.OffsetParameters()
    params.voxelSize = step
    params.signDetectionMode = mr.SignDetectionMode.WindingRule
    level = mr.offsetMesh(mrn.meshFromFacesVerts(faces.astype(np.int32), vertices.astype(np.float32)), 0.0, params)
    ctx.check_cancel()

    ctx.progress(0.6, "decimating")
    meshops.decimate_mesh(level, len(faces))
    ctx.check_cancel()

    ctx.progress(0.8, "separating pieces")
    pieces = _to_manifold(mrn.getNumpyVerts(level), mrn.getNumpyFaces(level.topology)).decompose()
    pieces.sort(key=lambda piece: -piece.volume())
    body = pieces[0]
    thick = [piece for piece in pieces[1:] if 2 * piece.volume() / piece.surface_area() >= THIN * height]
    body_vertices, body_faces = _from_manifold(body)
    near = []
    for piece in thick:
        piece_vertices = _from_manifold(piece)[0]
        squared = igl.point_mesh_squared_distance(piece_vertices, body_vertices, body_faces)[0]
        if np.sqrt(squared.min()) <= FAR * np.ptp(piece_vertices, axis=0).max():
            near.append((piece, piece_vertices))
    ctx.check_cancel()

    ctx.progress(0.85, "attaching pieces")
    gaps = []
    while near:
        body_vertices, body_faces = _from_manifold(body)
        best = None
        for index, (piece, piece_vertices) in enumerate(near):
            squared, _, closest = igl.point_mesh_squared_distance(piece_vertices, body_vertices, body_faces)
            nearest = int(np.argmin(squared))
            if best is None or squared[nearest] < best[0]:
                best = (squared[nearest], index, closest[nearest] - piece_vertices[nearest])
        squared_gap, index, direction = best
        gap = float(np.sqrt(squared_gap))
        body = body + near.pop(index)[0].translate(tuple(direction * (gap + OVERLAP * step) / gap))
        gaps.append(round(gap, 6))
        ctx.check_cancel()
    vertices, faces = _from_manifold(body)

    ctx.progress(0.95, "writing")
    path = ctx.dir / "mesh.glb"
    meshops.write_glb(path, vertices, faces, meshops.welded_normals(vertices, faces))
    return {
        "mesh": str(path),
        "removed_thin_pieces": len(pieces) - 1 - len(thick),
        "removed_far_pieces": len(thick) - len(gaps),
        "attached_pieces": len(gaps),
        "attached_gaps": gaps,
        "vertices": len(vertices),
        "faces": len(faces),
    }
