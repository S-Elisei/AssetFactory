"""Stage unwrap: welds the vertices of equal positions of a mesh and UV-unwraps it with xatlas into a GLB with UVs and
welded vertex normals, plus a UV layout image. Positions and faces are kept."""
import meshops
import numpy as np

# Atlas resolution in texels; also the size of the layout image.
ATLAS_SIZE = 2048


def _unwrap(vertices, faces):
    """UV-unwraps the mesh with xatlas. Returns (vmap, faces, uv): the unwrapped mesh has vertices[vmap] as vertices,
    `faces` indexes them, `uv` is (len(vmap), 2) in [0, 1]."""
    import xatlas

    chart = xatlas.ChartOptions()
    chart.max_chart_area = 0
    chart.max_boundary_length = 0
    chart.normal_deviation_weight = 2
    chart.roundness_weight = 0.01
    chart.straightness_weight = 6
    chart.normal_seam_weight = 4
    chart.texture_seam_weight = 0.5
    chart.max_cost = 16
    chart.max_iterations = 1
    chart.use_input_mesh_uvs = False
    chart.fix_winding = False
    pack = xatlas.PackOptions()
    pack.max_chart_size = 0
    pack.padding = 4
    pack.texels_per_unit = 0
    pack.resolution = ATLAS_SIZE
    pack.bilinear = True
    pack.blockAlign = False
    pack.bruteForce = False
    pack.create_image = False
    pack.rotate_charts_to_axis = True
    pack.rotate_charts = True
    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.uint32))
    atlas.generate(chart, pack)
    vmap, faces, uv = atlas[0]
    return vmap.astype(np.int64), faces.astype(np.int64), uv.astype(np.float32)


def _edge_table(faces):
    """Returns (keys, face_of_edge): one entry per face edge, `keys` identical for the two directions of an edge."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    keys = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    return keys, np.tile(np.arange(len(faces)), 3)


def _uv_charts(faces):
    """Returns (chart count, chart id of every face): charts are the faces connected through shared edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keys, face_of_edge = _edge_table(faces)
    order = np.argsort(keys, kind="stable")
    same = keys[order][1:] == keys[order][:-1]
    first, second = face_of_edge[order][:-1][same], face_of_edge[order][1:][same]
    graph = coo_matrix((np.ones(len(first), np.int8), (first, second)), shape=(len(faces), len(faces)))
    return connected_components(graph, directed=False)


def _uv_layout_image(uv, faces, count, chart):
    """Returns an ATLAS_SIZE x ATLAS_SIZE RGB image of the UV layout: every chart (`count` charts, `chart` the chart id
    of every face) filled with its own color on black, v pointing up."""
    from PIL import Image, ImageColor, ImageDraw

    colors = [ImageColor.getrgb(f"hsl({int(360 * (i * 0.61803398875 % 1))}, 75%, {45 + 15 * (i % 3)}%)")
              for i in range(count)]
    points = np.asarray(uv, np.float64)[faces] * [ATLAS_SIZE, -ATLAS_SIZE] + [0, ATLAS_SIZE]
    image = Image.new("RGB", (ATLAS_SIZE, ATLAS_SIZE))
    draw = ImageDraw.Draw(image)
    for polygon, c in zip(points.tolist(), chart.tolist()):
        draw.polygon([tuple(p) for p in polygon], fill=colors[c])
    return image


def run(ctx, mesh):
    vertices, faces, _ = meshops.load_input_mesh(mesh)

    ctx.progress(0.05, "unwrapping")
    vertices, faces = meshops.weld(vertices, faces)
    vmap, faces, uv = _unwrap(vertices, faces)
    vertices = vertices[vmap]
    ctx.check_cancel()

    ctx.progress(0.85, "writing")
    mesh_path, layout_path = ctx.dir / "mesh.glb", ctx.dir / "uv_layout.png"
    meshops.write_glb(mesh_path, vertices, faces, meshops.welded_normals(vertices, faces), uv)
    charts, chart = _uv_charts(faces)
    _uv_layout_image(uv, faces, charts, chart).save(layout_path)
    return {"mesh": str(mesh_path), "uv_layout": str(layout_path)}
