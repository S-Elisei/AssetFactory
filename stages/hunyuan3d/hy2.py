"""Stage hy2: the raw mesh of one object from an RGBA image with Hunyuan3D-DiT v2.0 turbo, written as `raw.glb`."""
import hunyuan3d_common as common
from PIL import Image

REPO = "tencent/Hunyuan3D-2"
KEEP_LOADED = False
# System RAM in GB that the Worker needs to start for this stage. Guessed.
RAM_GB = 4.0
DIRECTORY = "hunyuan3d-dit-v2-0-turbo"


def download():
    common.download(REPO, DIRECTORY)


def load():
    return common.load(REPO, DIRECTORY)


def run(ctx, image, steps, guidance_scale, octree_resolution, num_chunks, seed):
    return common.generate(ctx, ctx.model, Image.open(image), steps, guidance_scale, octree_resolution, num_chunks,
                           seed)
