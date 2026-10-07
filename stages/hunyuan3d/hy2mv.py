"""Stage hy2mv: the raw mesh of one object from its RGBA views with Hunyuan3D-DiT v2 mv turbo, written as `raw.glb`.
`front` is required; `left`, `back` and `right` are image paths or None."""
import hunyuan3d_common as common
from PIL import Image

REPO = "tencent/Hunyuan3D-2mv"
KEEP_LOADED = False
DIRECTORY = "hunyuan3d-dit-v2-mv-turbo"


def download():
    common.download(REPO, DIRECTORY)


def load():
    return common.load(REPO, DIRECTORY)


def run(ctx, front, left, back, right, steps, guidance_scale, octree_resolution, num_chunks, seed):
    views = {tag: Image.open(path) for tag, path in (("front", front), ("left", left), ("back", back),
                                                     ("right", right)) if path is not None}
    return common.generate(ctx, ctx.model, views, steps, guidance_scale, octree_resolution, num_chunks, seed)
