"""Modal app of the hunyuan3d-2.1 cloud stage: the raw mesh of one object from an image with Hunyuan3D-Shape v2.1 (the
flow-matching DiT, the VAE decoder and marching cubes), as `raw.glb`. In one container: background removal with the
`bgremove` stage code (BiRefNet), then the shape. The mesh gets no face reduction, floater removal, UV unwrap or
texture.

The weights are in the Modal Volume `VOLUME`, mounted at WEIGHTS: the Hugging Face cache of the repositories and the
safetensors file WEIGHTS_FILE that `download_weights` writes from the repository's checkpoint, all floating tensors in
fp16. The models are built without weights and the tensors are loaded straight onto the GPU. Calls: see the package
docstring."""
import time
from pathlib import Path
from queue import Full

import modal

APP_NAME = "assetfactory-hunyuan3d21"
VOLUME = "assetfactory-hunyuan3d21-weights"
WEIGHTS_APP_NAME = f"{APP_NAME}-weights"
CLASS = "Hunyuan3D21"
# GPU type, CPU cores, memory in MiB, seconds a container stays up without a call, and seconds a call or a container
# start may take. The last four are guessed.
GPU = "L40S"
CPU = 2
MEMORY_MIB = 32768
SCALEDOWN_SECONDS = 60
TIMEOUT_SECONDS = 900

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "/weights"
SOURCE = "/opt/hy3d21"
SHARED = "/opt/af"
HUNYUAN_COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
REPO = "tencent/Hunyuan3D-2.1"
DIRECTORY = "hunyuan3d-dit-v2-1"
# Written by download_weights, read by load: the checkpoint's tensors named `<group>.<name>` for the groups `model`,
# `vae` and `conditioner`.
WEIGHTS_FILE = Path(WEIGHTS) / "hunyuan3d21.safetensors"
# Points of the shape volume that the VAE decoder evaluates per batch. Guessed.
NUM_CHUNKS = 20000
# Seconds between two progress messages that are not forced, and seconds a put of a message may take. Guessed.
PROGRESS_INTERVAL = 0.5
PROGRESS_TIMEOUT = 5.0


def reporter(queue):
    """Returns `report(fraction, message, force=False)`, which puts the tuple (fraction, message) on the modal.Queue
    `queue`; a message that is not forced is dropped when the previous message was put less than PROGRESS_INTERVAL
    seconds earlier, and a message whose put raises a Modal error or does not finish within PROGRESS_TIMEOUT seconds is
    dropped."""
    last = [float("-inf")]

    def report(fraction, message, force=False):
        now = time.monotonic()
        if force or now - last[0] >= PROGRESS_INTERVAL:
            last[0] = now
            try:
                queue.put((fraction, message), timeout=PROGRESS_TIMEOUT)
            except (modal.Error, Full):
                pass

    return report


def _silent(fraction, message, force=False):
    pass


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.46.0", "diffusers==0.30.0", "accelerate==1.1.1", "pytorch-lightning==1.9.5",
                 "huggingface-hub==0.30.2", "hf-xet==1.0.3", "safetensors==0.4.4", "numpy==1.24.4", "scipy==1.14.1",
                 "einops==0.8.0", "opencv-python-headless==4.10.0.84", "imageio==2.36.0", "scikit-image==0.24.0",
                 "kornia==0.8.3", "trimesh==4.4.7", "pymeshlab==2022.2.post3", "omegaconf==2.3.0", "pyyaml==6.0.2",
                 "tqdm==4.66.5", "timm==1.0.15", "torchdiffeq==0.2.5", "pillow==10.4.0", "setuptools==75.8.0")
    .env({"HF_HOME": f"{WEIGHTS}/hf", "PYTHONPATH": f"{SOURCE}/hy3dshape:{SHARED}"})
    .run_commands(
        f"git clone -q https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git {SOURCE} && "
        f"git -C {SOURCE} checkout -q {HUNYUAN_COMMIT}",
    )
    .add_local_file(ROOT / "stages" / "triposg" / "bgremove.py", f"{SHARED}/bgremove.py")
    .add_local_file(ROOT / "shared" / "mapped.py", f"{SHARED}/mapped.py")
    .add_local_file(ROOT / "worker" / "context.py", f"{SHARED}/context.py")
)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)
weights_app = modal.App(WEIGHTS_APP_NAME, image=image)


@weights_app.function(volumes={WEIGHTS: volume}, memory=MEMORY_MIB, timeout=3600)
def download_weights(hf_token):
    """Fetches the files that `load` reads into the Volume and writes WEIGHTS_FILE; files already written are kept. The
    file holds the floating tensors of the checkpoint in fp16; of the group `vae` it holds only the tensors that have a
    parameter or buffer of that name in the VAE."""
    import os

    import bgremove
    import mapped
    import torch
    import yaml
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from hy3dshape.pipelines import instantiate_from_config

    os.environ["HF_TOKEN"] = hf_token
    config = yaml.safe_load(Path(hf_hub_download(REPO, f"{DIRECTORY}/config.yaml")).read_text(encoding="utf-8"))
    checkpoint = hf_hub_download(REPO, f"{DIRECTORY}/model.fp16.ckpt")
    bgremove.download()
    if not WEIGHTS_FILE.exists():
        with init_empty_weights():
            vae_names = set(instantiate_from_config(config["vae"]).state_dict())
        groups = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        mapped.save_weights(WEIGHTS_FILE, {
            f"{group}.{name}": (tensor.to(torch.float16) if tensor.is_floating_point() else tensor).contiguous()
            for group, state in groups.items() for name, tensor in state.items()
            if group != "vae" or name in vae_names})
    volume.commit()


def _load(module, tensors):
    """Deletes every top-level submodule of `module` that holds none of `tensors` (named as the state dict of `module`),
    then loads `tensors` as the parameters and buffers of `module`, every one of which must be among them."""
    held = {key.split(".")[0] for key in tensors}
    for name, _ in list(module.named_children()):
        if name not in held:
            delattr(module, name)
    module.load_state_dict(tensors, assign=True)


@app.cls(gpu=GPU, cpu=CPU, memory=MEMORY_MIB, max_containers=1, volumes={WEIGHTS: volume}, timeout=TIMEOUT_SECONDS,
         scaledown_window=SCALEDOWN_SECONDS, enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True},
         env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
class Hunyuan3D21:
    @modal.enter(snap=True)
    def load(self):
        """Loads the shape pipeline and BiRefNet onto the GPU, then runs one short generation. The BiRefNet tensors
        are copied to the GPU and their memory maps released."""
        import bgremove
        import torch
        import yaml
        from accelerate import init_empty_weights
        from huggingface_hub import hf_hub_download
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config
        from safetensors.torch import load_file

        config = yaml.safe_load(Path(hf_hub_download(REPO, f"{DIRECTORY}/config.yaml")).read_text(encoding="utf-8"))
        tensors = load_file(WEIGHTS_FILE, device="cuda")

        def build(group, load):
            with init_empty_weights():
                module = instantiate_from_config(config[group])
            load(module, {key[len(group) + 1:]: tensor for key, tensor in tensors.items()
                          if key.startswith(group + ".")})
            return module

        def strict(module, state):
            module.load_state_dict(state, assign=True)

        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline(
            vae=build("vae", _load),
            model=build("model", strict),
            scheduler=instantiate_from_config(config["scheduler"]),
            conditioner=build("conditioner", strict),
            image_processor=instantiate_from_config(config["image_processor"]),
            device="cuda",
            dtype=torch.float16,
        )
        del tensors

        self.remover = bgremove.load().to("cuda")
        for tensor in (*self.remover.parameters(), *self.remover.buffers()):
            if hasattr(tensor, "host"):
                del tensor.host
        demo = Path(SOURCE) / "assets" / "demo.png"
        self._generate(demo.read_bytes(), {"steps": 2, "guidance_scale": 5.0, "octree_resolution": 128, "seed": 0},
                       _silent)

    @modal.enter(snap=False)
    def start(self):
        """Names this container: an id, and the Unix time at which it began to take calls."""
        import uuid

        self.container = {"container": uuid.uuid4().hex, "container_started": time.time()}

    @modal.method()
    def run(self, image, params, progress):
        from context import InputError

        try:
            result = self._generate(image, params, reporter(progress))
        except InputError:
            raise
        except Exception:
            modal.experimental.stop_fetching_inputs()
            raise
        return {**result, **self.container}

    def _generate(self, data, params, report):
        import io

        import bgremove
        import torch
        from context import InputError
        from PIL import Image

        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            source = Image.open(io.BytesIO(data))
            source.load()
        except OSError:
            raise InputError("image: the file is not a readable image; send a PNG, JPEG or WEBP")
        report(0.0, "removing the background", True)
        rgba = bgremove.remove_background(self.remover, source)

        steps = params["steps"]

        def on_step(step, _timestep, _outputs):
            report(0.05 + 0.65 * (step + 1) / steps, f"sampling step {step + 1}/{steps}")
            if step + 1 == steps:
                report(0.7, "extracting the surface", True)

        report(0.05, "encoding the image", True)
        mesh = self.pipeline(image=rgba, num_inference_steps=params["steps"], guidance_scale=params["guidance_scale"],
                             octree_resolution=params["octree_resolution"], num_chunks=NUM_CHUNKS,
                             generator=torch.Generator().manual_seed(params["seed"]), output_type="trimesh",
                             enable_pbar=False, callback=on_step, callback_steps=1)[0]
        report(0.95, "writing", True)
        glb = mesh.export(file_type="glb", include_normals=False)
        torch.cuda.empty_cache()
        return {"raw.glb": glb, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "seconds": round(time.monotonic() - started, 3)}
