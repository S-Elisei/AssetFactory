"""Stage decimate: decimates a mesh with QEM edge collapse (MeshLib, topology kept) down to `target_faces` and writes it
as a GLB with welded vertex normals. A mesh with no more faces is written unchanged."""
import meshops

KEEP_LOADED = False


def run(ctx, mesh, target_faces):
    vertices, faces, _ = meshops.load_glb(mesh)

    ctx.progress(0.0, "decimating")
    vertices, faces = meshops.decimate(vertices, faces, target_faces)
    ctx.check_cancel()

    ctx.progress(0.9, "writing")
    path = ctx.dir / "mesh.glb"
    meshops.write_glb(path, vertices, faces, meshops.welded_normals(vertices, faces))
    return {"mesh": str(path), "vertices": len(vertices), "faces": len(faces)}
