"""The Modal apps of the cloud stages, one module each: `trellis2`, `hunyuan3d21`, `unitex`. A module imports `modal` and
the standard library at the top level; everything else is imported inside the functions that run in the container.

Every module defines `APP_NAME`, `app` (the `modal.App` that holds the class named below, with the remote method
`run`), `GPU`, `CPU`, `MEMORY_MIB` and `SCALEDOWN_SECONDS` (the container's resources and the seconds it stays up
without a call), and `WEIGHTS_APP_NAME` and `weights_app`, a second `modal.App` with the same image and Volume that holds
only the function `download_weights`.

Install: `core.cli install <app>` runs `with module.weights_app.run(): module.download_weights.remote(hf_token)`, which
fills the app's Modal Volume from Hugging Face with the token `hf_token` (a str) and returns None, and then
`module.app.deploy()`. Running the weights app registers no class that takes a memory snapshot, so no snapshot is taken
while the Volume is empty. Install can be repeated to complete an interrupted download.

Call: `modal.Cls.from_name(APP_NAME, CLASS)().run.spawn(..., progress)`, `.get()` on the result. Inputs are file
contents as `bytes` and `params`, a dict of JSON-able values of exactly the keys listed here; `run` checks no param.
`progress` is the `modal.Queue` of the call, the last argument of `run`.

| module       | class        | `run` arguments                  | params                                                       |
|--------------|--------------|----------------------------------|--------------------------------------------------------------|
| trellis2     | Trellis2     | `image`, `params`, `progress`    | `resolution` (512, 1024 or 1536), `steps`, `decimation_target`, `seed` |
| hunyuan3d21  | Hunyuan3D21  | `image`, `params`, `progress`    | `steps`, `guidance_scale`, `octree_resolution`, `seed`       |
| unitex       | UniTEX       | `mesh`, `image`, `params`, `progress` | `delight` (bool), `texture_size`, `seed`                     |

`image` is the file of an image PIL reads. An image with an alpha channel on which at least 5 percent of the pixels are
at alpha 0 is used as it is; for any other image the container removes the background with BiRefNet. `mesh` (unitex) is
the GLB of a triangle mesh with non-overlapping UVs; `texture_size` is the side in texels at which the overlap is
measured.

Progress: the CloudQueue creates one queue per call with `with modal.Queue.ephemeral() as progress:`, passes it to
`spawn` and, while the call runs, loops: `progress.get_many(100, block=False)` returns the tuples `(fraction, message)`
put so far (a list, empty when there are none; `fraction` is a float in 0..1 of the call, `message` a str), then
`call.get(timeout=1)`; the builtin `TimeoutError` means the call is not finished. After `get` returns, one more
`get_many` takes the messages put last. Leaving the `with` block deletes the queue; the queue is created and deleted
by the client only. The container puts one message at the start of each phase (background removal, sampling,
remeshing or surface extraction, views, delight, upscaling, writing) and, for the sampling steps, at most one message
per `PROGRESS_INTERVAL` seconds (0.5); a queue holds 5000 messages.

Result: a dict. Every `bytes` value is an output file, named by its key; every other value is JSON-able.
- trellis2: `raw.glb` (positions and faces, Y up); `resolution` (the voxel resolution used), `vertices`, `faces`.
- hunyuan3d21: `raw.glb`; `vertices`, `faces`.
- unitex: `cameras.json`, in the format of the `bake` stage and shared by both view sets; `lit_view_0.png` ..
  `lit_view_5.png`, the views as they are generated; with `delight` true also `delit_view_0.png` .. `delit_view_5.png`,
  the same views after the delight pass. Each view is an RGBA PNG whose alpha is the silhouette of the mesh; view `n`
  belongs to entry `n` of `cameras`. The input mesh is the mesh `bake` uses; no mesh is returned.
- every app: `vram_peak_gb`, the peak CUDA memory allocated during the call; `seconds`, the time of the call inside the
  container; `container`, a str that names the container that ran the call; `container_started`, the Unix time at which
  that container began to take calls. `seconds` leaves out the container's start and the idle seconds before its stop.

Errors: `context.InputError` (the module `context` of `worker/context.py`, which the container has under the same
name) is raised in the container for an unreadable image and, in unitex, for an unusable mesh; with `<root>/worker` on
its path the client receives it as that class. Any other exception propagates as it is; the client receives it as
itself when the class is importable there, else as `modal.exception.ExecutionError` whose message holds the remote
traceback. After any exception other than InputError the container takes no further call and the next call starts a
new container.

A call that runs longer than `TIMEOUT_SECONDS` raises `modal.exception.FunctionTimeoutError`, which is not the builtin
`TimeoutError` of `get(timeout=...)`.

Cancel: `FunctionCall.cancel()` raises `modal.exception.InputCancellation` inside `run`. `run` does not catch it; the
call ends as cancelled and the container keeps taking calls.

Container lifetime: at most one container, which runs one call at a time and stops `SCALEDOWN_SECONDS` after its last
call. A new container starts from a memory snapshot (CPU and GPU) taken after the models are loaded.

Views of unitex: the camera of view `n` has `c2w` rigid in the frame of the input mesh and `left = -h`, `right = h`,
`bottom = -h`, `top = h` for the half side `h` of the view in the units of the input mesh; row 0 of a view is its top.
"""
