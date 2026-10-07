"""The Modal apps of the cloud stages, one module each: `trellis2`, `hunyuan3d21`, `unitex`, and the code that they
share. A module imports `modal`, the standard library and names from this package at the top level; everything else is
imported inside the functions that run in the container.

A module defines `APP_NAME`; `app`, the `modal.App` that holds the class `CLASS`, whose remote method `run` is the
stage; `GPU`, `CPU`, `MEMORY_MIB` and `SCALEDOWN_SECONDS`; and `WEIGHTS_APP_NAME` and `weights_app`, a `modal.App` with
the same image and Volume that holds only the function `download_weights(hf_token)`, which fills the Volume with the
weights and returns None. Install runs `download_weights` in an ephemeral run of `weights_app` and then deploys `app`.
The class takes at most one container and one call at a time; the container stops `SCALEDOWN_SECONDS` after its last
call and starts from a memory snapshot taken after the models are loaded.

| module       | class        | `run` arguments                       | params                                                              |
|--------------|--------------|---------------------------------------|---------------------------------------------------------------------|
| trellis2     | Trellis2     | `image`, `params`, `progress`         | `resolution` (512, 1024 or 1536), `steps`, `decimation_target`, `seed` |
| hunyuan3d21  | Hunyuan3D21  | `image`, `params`, `progress`         | `steps`, `guidance_scale`, `octree_resolution`, `seed`              |
| unitex       | UniTEX       | `mesh`, `image`, `params`, `progress` | `delight` (bool), `texture_size`, `seed`                            |

`image` and `mesh` are file contents as `bytes`. `image` is the file of an image PIL reads; the container removes its
background with BiRefNet unless it is already cut out. `mesh` (unitex) is the GLB of a triangle mesh with
non-overlapping UVs; `texture_size` is the side in texels at which the overlap is measured. `params` is a dict of
JSON-able values of exactly the listed keys; `run` checks none of them.

`progress` is a `modal.Queue`. `run` puts on it the tuples `(fraction, message)`, `fraction` a float in 0..1 of the call
and `message` a str, at the start of each phase and at the sampling steps. A put that fails is dropped. The last
message of a call is `DONE`: it is put after the last piece of work of `run`, immediately before `run` returns.

`run` returns a dict. Every `bytes` value is an output file, named by its key; every other value is JSON-able.
- trellis2: `raw.glb` (positions and faces, Y up); `resolution` (the voxel resolution used), `vertices`, `faces`.
- hunyuan3d21: `raw.glb`; `vertices`, `faces`.
- unitex: `cameras.json`, in the format of the `bake` stage and shared by both view sets; `lit_view_0.png` ..
  `lit_view_5.png`, the views as they are generated; with `delight` true also `delit_view_0.png` .. `delit_view_5.png`,
  the same views after the delight pass. Each view is an RGBA PNG whose alpha is the silhouette of the mesh; view `n`
  belongs to entry `n` of `cameras`. The input mesh is the mesh `bake` uses; no mesh is returned.
- every app: `seconds`, the time of the call inside the container.

`run` raises `context.InputError` for an unreadable image and, in unitex, for an unusable mesh. After any other
exception the container takes no further call.

Views of unitex: the camera of view `n` has `c2w` rigid in the frame of the input mesh and `left = -h`, `right = h`,
`bottom = -h`, `top = h` for the half side `h` of the view in the units of the input mesh; row 0 of a view is its top.
"""
import time

import modal

# The last message of a call.
DONE = (1.0, "done")
# Seconds between two progress messages that are not forced. Guessed.
PROGRESS_INTERVAL = 0.5


def reporter(queue):
    """Returns `report(fraction, message, force=False)`, which puts the tuple (fraction, message) on the modal.Queue
    `queue`; a message that is not forced is dropped when the previous message was put less than PROGRESS_INTERVAL
    seconds earlier, and a message whose put raises a Modal error is dropped."""
    last = [float("-inf")]

    def report(fraction, message, force=False):
        now = time.monotonic()
        if force or now - last[0] >= PROGRESS_INTERVAL:
            last[0] = now
            try:
                queue.put((fraction, message))
            except modal.Error:
                pass

    return report


def silent(fraction, message, force=False):
    """A `report` that drops every message."""


def run_call(generate, progress, *arguments, last=None):
    """Calls `generate(*arguments, report)` with the reporter of the modal.Queue `progress`, then `last()` when given,
    puts DONE and returns the result of `generate`. After an exception other than InputError the container takes no
    further call."""
    from context import InputError

    report = reporter(progress)
    try:
        result = generate(*arguments, report)
    except InputError:
        raise
    except Exception:
        modal.experimental.stop_fetching_inputs()
        raise
    if last is not None:
        last()
    report(*DONE, True)
    return result


def read_image(data):
    """Returns the loaded PIL image of the file contents `data`. Raises InputError when it is not a readable image."""
    import io

    from context import InputError
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except OSError:
        raise InputError("image: the file is not a readable image; send a PNG, JPEG or WEBP")
    return image


def load_strict(module, tensors):
    """Deletes every top-level submodule of `module` that holds none of `tensors` (named as the state dict of `module`),
    then loads `tensors` as the parameters and buffers of `module`, every one of which must be among them."""
    held = {key.split(".")[0] for key in tensors}
    for name, _ in list(module.named_children()):
        if name not in held:
            delattr(module, name)
    module.load_state_dict(tensors, assign=True)
