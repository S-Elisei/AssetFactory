"""Modal app of the UniTEX views cloud stage: six orthographic views of a UV-mapped mesh from one reference image with
UniTEX's multiview texturing (FLUX.1-dev with the UniTEX texture LoRA, optionally followed by the delight LoRA pass),
upscaled with TSD-SR, and the cameras of the views in the frame of the input mesh. UniTEX's LTM and its reprojection
are not run. In one container: background removal of the reference with the `bgremove` stage code (BiRefNet), then
the views. The input mesh is checked first, before any model runs, with the mesh code of `shared/meshops.py`.

The views are rendered and generated for the mesh moved to v' = (v - c) / s, with c the center of the bounding box of
the vertices that the faces use and s its largest extent divided by 2 * GEOMETRY_SCALE. The returned camera of view `n`
has the rotation of the camera of the moved mesh and the translation s * t' + c, which makes it rigid in the units of
the input mesh; its half side is s divided by CAMERA_SCALE.

The weights are in the Modal Volume `VOLUME`, mounted at WEIGHTS: the Hugging Face cache of the repositories, the
bf16 copy of the FLUX VAE in CONVERTED and the TSD-SR files in TSDSR. The models are built without weights and their
tensors are loaded straight onto the GPU, except the two SD3 models that TSD-SR builds itself. Calls: see the package
docstring."""
import time
from pathlib import Path
from queue import Full

import modal

APP_NAME = "assetfactory-unitex"
VOLUME = "unitex-weights"
CLASS = "UniTEX"
# GPU type, CPU cores, memory in MiB, seconds a container stays up without a call, and seconds a call or a container
# start may take. The last four are guessed.
GPU = "L40S"
CPU = 4
MEMORY_MIB = 24576
SCALEDOWN_SECONDS = 60
TIMEOUT_SECONDS = 900

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "/weights"
SOURCE = "/opt/unitex"
SHARED = "/opt/af"
UNITEX_COMMIT = "affa1e29e665670dbdfd13ee4a3a68a45942df47"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"

FLUX_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_FILES = ["model_index.json", "scheduler/*", "transformer/*", "vae/*"]
SD3_REPO = "stabilityai/stable-diffusion-3-medium-diffusers"
SD3_FILES = ["model_index.json", "transformer/*", "vae/*"]
UNITEX_REPO = "lyxun/UniTEX"
# The LoRA files of the texture pass and of the delight pass, in the order of ADAPTERS.
LORAS = ["mv_lora_weights.safetensors", "delight_lora_weights.safetensors"]
ADAPTERS = ["texture", "delight"]
CONVERTED = Path(WEIGHTS) / "unitex"
VAE_WEIGHTS = CONVERTED / "vae.safetensors"
TSDSR = Path(WEIGHTS) / "tsdsr"
TSDSR_DRIVE = "https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI"
# File name and folder in TSDSR of each TSD-SR file.
TSDSR_FILES = {"transformer.safetensors": "checkpoint/tsdsr", "vae.safetensors": "checkpoint/tsdsr",
               "prompt_embeds.pt": "dataset/default", "pool_embeds.pt": "dataset/default"}

# Largest extent of the normalised mesh divided by 2, side in pixels of the geometry renders and of the views before
# upscaling, intrinsics scale of the orthographic cameras (the image plane covers [-1, 1] of the normalised frame), and
# sampling steps and guidance scale of both FLUX passes. Documented.
GEOMETRY_SCALE = 0.95
VIEW_SIZE = 512
CAMERA_SCALE = 1.0
STEPS = 28
GUIDANCE_SCALE = 3.5
# Side in pixels of the reference image before the foreground is cropped and placed, the share of that side the
# foreground fills, the side of the image given to FLUX, and the background color. Documented.
REFERENCE_SIZE = 1024
REFERENCE_FILL = 0.95
CONDITION_SIZE = 512
BACKGROUND = "grey"
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


image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libsm6", "libxext6",
                 "libxrender1", "libegl1")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("xformers==0.0.28.post1", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("transformers==4.52.4", "diffusers==0.32.2", "peft==0.15.2", "accelerate==1.7.0",
                 "safetensors==0.5.3", "huggingface_hub[hf_xet]==0.34.4", "numpy==1.26.4", "scipy==1.14.1",
                 "pillow==10.4.0", "opencv-python-headless==4.10.0.84", "imageio==2.37.0", "trimesh==3.20.2",
                 "pymeshlab==2023.12.post3", "open3d==0.19.0", "gpytoolbox==0.3.3", "einops==0.8.0", "tqdm==4.67.1",
                 "timm==1.0.15", "kornia==0.8.0", "gdown==5.2.0", "setuptools==75.8.0", "wheel==0.45.1",
                 "ninja==1.13.0")
    .env({"TORCH_CUDA_ARCH_LIST": "8.9", "CC": "gcc", "CXX": "g++", "HF_HOME": f"{WEIGHTS}/hf",
          "PYTHONPATH": f"{SOURCE}:{SHARED}"})
    .run_commands(
        f"git clone -q https://github.com/NVlabs/nvdiffrast.git /opt/ext/nvdiffrast && "
        f"git -C /opt/ext/nvdiffrast checkout -q {NVDIFFRAST_COMMIT}",
        "pip install --no-build-isolation /opt/ext/nvdiffrast",
        f"git clone -q https://github.com/YixunLiang/UniTEX.git {SOURCE} && "
        f"git -C {SOURCE} checkout -q {UNITEX_COMMIT}",
    )
    .add_local_file(ROOT / "stages" / "triposg" / "bgremove.py", f"{SHARED}/bgremove.py")
    .add_local_file(ROOT / "shared" / "mapped.py", f"{SHARED}/mapped.py")
    .add_local_file(ROOT / "shared" / "meshops.py", f"{SHARED}/meshops.py")
    .add_local_file(ROOT / "worker" / "context.py", f"{SHARED}/context.py")
)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(volumes={WEIGHTS: volume}, memory=MEMORY_MIB, timeout=3600)
def download_weights(hf_token):
    """Fetches the files that `load` reads into the Volume and writes the bf16 copy of the FLUX VAE (VAE_WEIGHTS); files
    already fetched or written are kept. The TSD-SR files come from the Google Drive folder TSDSR_DRIVE."""
    import os
    import shutil
    import subprocess
    import tempfile

    import bgremove
    import mapped
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file

    os.environ["HF_TOKEN"] = hf_token
    flux = Path(snapshot_download(FLUX_REPO, allow_patterns=FLUX_FILES))
    if not VAE_WEIGHTS.exists():
        CONVERTED.mkdir(parents=True, exist_ok=True)
        mapped.save_weights(VAE_WEIGHTS, {key: tensor.to(torch.bfloat16) for key, tensor in
                                          load_file(flux / "vae" / "diffusion_pytorch_model.safetensors").items()})
    snapshot_download(SD3_REPO, allow_patterns=SD3_FILES)
    for name in LORAS:
        hf_hub_download(UNITEX_REPO, name)
    bgremove.download()
    if not all((TSDSR / folder / name).exists() for name, folder in TSDSR_FILES.items()):
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(["gdown", "--folder", TSDSR_DRIVE, "-O", temporary], check=True)
            for name, folder in TSDSR_FILES.items():
                (TSDSR / folder).mkdir(parents=True, exist_ok=True)
                shutil.copy(next(Path(temporary).rglob(name)), TSDSR / folder / name)
    volume.commit()


def _model(factory, files):
    """Returns the model that `factory` builds without weights, with the tensors of the safetensors `files` loaded
    straight onto the GPU."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    with init_empty_weights():
        model = factory()
    tensors = {}
    for file in files:
        tensors.update(load_file(file, device="cuda"))
    model.load_state_dict(tensors, assign=True)
    return model.to("cuda").eval()


def _reference(rgba):
    """Returns the RGB image of the RGBA `rgba` that FLUX takes as reference: the image scaled to REFERENCE_SIZE, its
    foreground (alpha above 0) cropped, scaled to fill REFERENCE_FILL of the side and centered on BACKGROUND, then
    scaled to CONDITION_SIZE."""
    import numpy as np
    from PIL import Image

    rgba = rgba.resize((REFERENCE_SIZE, REFERENCE_SIZE))
    alpha = rgba.getchannel("A")
    rows, columns = np.nonzero(np.asarray(alpha))
    box = (int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max()))
    height, width = box[3] - box[1], box[2] - box[0]
    ratio = min(REFERENCE_SIZE * REFERENCE_FILL / height, REFERENCE_SIZE * REFERENCE_FILL / width)
    new_height, new_width = int(height * ratio), int(width * ratio)
    left, top = int((REFERENCE_SIZE - new_width) / 2), int((REFERENCE_SIZE - new_height) / 2)
    canvas = Image.new("RGB", (REFERENCE_SIZE, REFERENCE_SIZE), BACKGROUND)
    canvas.paste(rgba.convert("RGB").crop(box).resize((new_width, new_height)),
                 (left, top, left + new_width, top + new_height), alpha.crop(box).resize((new_width, new_height)))
    return canvas.resize((CONDITION_SIZE, CONDITION_SIZE))


def _control(normal, ccm):
    """Returns the FLUX control image: the mean of the grids `normal` and `ccm` (PIL images of 2 rows by 3 columns of
    views, row by row front, right, top, back, left, bottom), with the bottom view turned by 180 degrees, as a strip of
    the six views side by side in the order front, left, right, back, top, bottom."""
    import numpy as np
    from PIL import Image

    mean = (0.5 * np.asarray(normal).reshape(2, VIEW_SIZE, 3, VIEW_SIZE, -1)
            + 0.5 * np.asarray(ccm).reshape(2, VIEW_SIZE, 3, VIEW_SIZE, -1)).astype(np.uint8)
    mean[1, :, 2] = mean[1, ::-1, 2, ::-1]
    views = mean.transpose(0, 2, 1, 3, 4).reshape(6, VIEW_SIZE, VIEW_SIZE, -1)[[0, 4, 1, 3, 2, 5]]
    return Image.fromarray(views.transpose(1, 0, 2, 3).reshape(VIEW_SIZE, 6 * VIEW_SIZE, -1))


def _grid(strip):
    """Returns the grid of the FLUX output `strip`, a strip of the six views side by side in the order front, left,
    right, back, top, bottom, as a PIL image of 2 rows by 3 columns of views, row by row front, right, top, back, left,
    bottom, with the bottom view turned by 180 degrees."""
    import numpy as np
    from PIL import Image

    views = np.array(strip).reshape(VIEW_SIZE, 6, VIEW_SIZE, -1)
    views[:, 5] = views[::-1, 5, ::-1]
    return Image.fromarray(views.transpose(1, 0, 2, 3)[[0, 2, 4, 3, 1, 5]].reshape(2, 3, VIEW_SIZE, VIEW_SIZE, -1)
                           .transpose(0, 2, 1, 3, 4).reshape(2 * VIEW_SIZE, 3 * VIEW_SIZE, -1))


@app.cls(gpu=GPU, cpu=CPU, memory=MEMORY_MIB, max_containers=1, volumes={WEIGHTS: volume}, timeout=TIMEOUT_SECONDS,
         scaledown_window=SCALEDOWN_SECONDS, enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True},
         env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
class UniTEX:
    @modal.enter(snap=True)
    def load(self):
        """Loads FLUX with both LoRAs, TSD-SR and BiRefNet onto the GPU. The BiRefNet tensors are copied to the GPU
        and their memory maps released."""
        import sys

        import bgremove
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
        from flux_piplines.texturing.pipeline import PBRFluxPipeline
        from huggingface_hub import hf_hub_download, snapshot_download
        from TSD_SR.sr_pipeline import TSDSRPipeline

        flux = Path(snapshot_download(FLUX_REPO, allow_patterns=FLUX_FILES))
        self.pipeline = PBRFluxPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(flux, subfolder="scheduler"),
            vae=_model(lambda: AutoencoderKL.from_config(AutoencoderKL.load_config(flux / "vae")),
                       [VAE_WEIGHTS]),
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            transformer=_model(lambda: FluxTransformer2DModel.from_config(FluxTransformer2DModel.load_config(
                flux / "transformer")), sorted((flux / "transformer").glob("diffusion_pytorch_model-*.safetensors"))),
        )
        for name, file in zip(ADAPTERS, LORAS):
            self.pipeline.load_lora_weights(hf_hub_download(UNITEX_REPO, file), adapter_name=name)
        self.pipeline.set_progress_bar_config(disable=True)

        sd3 = Path(snapshot_download(SD3_REPO, allow_patterns=SD3_FILES))
        sys.argv = ["tsdsr", "--pretrained_model_name_or_path", str(sd3), "--lora_dir",
                    str(TSDSR / "checkpoint" / "tsdsr"), "--embedding_dir", str(TSDSR / "dataset" / "default")]
        self.upscaler = TSDSRPipeline()

        self.remover = bgremove.load().to("cuda")
        for tensor in (*self.remover.parameters(), *self.remover.buffers()):
            if hasattr(tensor, "host"):
                del tensor.host

    @modal.enter(snap=False)
    def start(self):
        """Names this container: an id, and the Unix time at which it began to take calls."""
        import uuid

        self.container = {"container": uuid.uuid4().hex, "container_started": time.time()}

    @modal.method()
    def run(self, mesh, image, params, progress):
        from context import InputError

        try:
            result = self._generate(mesh, image, params, reporter(progress))
        except InputError:
            raise
        except Exception:
            modal.experimental.stop_fetching_inputs()
            raise
        return {**result, **self.container}

    def _generate(self, data, reference, params, report):
        import io
        import json
        import tempfile

        import bgremove
        import meshops
        import torch
        import trimesh
        from context import InputError
        from PIL import Image
        from TextureTools.texturetools.video.export_nvdiffrast_video import VideoExporter

        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        report(0.0, "checking the mesh", True)
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / "input.glb"
            input_path.write_bytes(data)
            try:
                vertices, faces, uv = meshops.load_glb(str(input_path))
            except Exception:
                raise InputError("mesh: the file is not a readable GLB; send a binary glTF (.glb) with a triangle mesh")
            if len(faces) == 0:
                raise InputError("mesh: the GLB contains no triangle mesh; send a binary glTF (.glb) with a triangle "
                                 "mesh")
            if uv is None:
                raise InputError("mesh: the GLB has no UV coordinates (TEXCOORD_0); run mesh_unwrap on it first and "
                                 "send its output")
            overlap = meshops.uv_overlap_texels(uv, faces, params["texture_size"])
            if overlap:
                raise InputError(f"mesh: {overlap} texels of the {params['texture_size']} x {params['texture_size']} "
                                 "texture lie inside more than one UV triangle; run mesh_unwrap on it first and send "
                                 "its output")
            try:
                source = Image.open(io.BytesIO(reference))
                source.load()
            except OSError:
                raise InputError("image: the file is not a readable image; send a PNG, JPEG or WEBP")

            report(0.05, "removing the background", True)
            condition = _reference(bgremove.remove_background(self.remover, source))
            report(0.1, "rendering the geometry", True)
            geometry_path = Path(folder) / "geometry.glb"
            trimesh.Trimesh(vertices, faces, process=False).export(str(geometry_path))
            exporter = VideoExporter()
            rendered = exporter.export_condition(
                str(geometry_path), geometry_scale=GEOMETRY_SCALE, n_views=6, n_rows=2, n_cols=3, H=VIEW_SIZE,
                W=VIEW_SIZE, scale=CAMERA_SCALE, perspective=False, orbit=False, background=BACKGROUND,
                return_image=True, return_camera=True)
            del exporter

        used = vertices[faces.reshape(-1)]
        low, high = used.min(axis=0), used.max(axis=0)
        center, unit = (low + high) / 2, float((high - low).max() / (2 * GEOMETRY_SCALE))
        half = unit / CAMERA_SCALE
        cameras = []
        for c2w in rendered["c2ws"].double().cpu().numpy():
            c2w[:3, 3] = c2w[:3, 3] * unit + center
            cameras.append({"c2w": c2w.tolist(), "left": -half, "right": half, "bottom": -half, "top": half})
        files = {"cameras.json": json.dumps({"cameras": cameras}).encode("utf-8")}

        generator = torch.Generator().manual_seed(params["seed"])
        passes = 2 if params["delight"] else 1

        def sampling(index, label):
            report(0.15 + 0.55 * index / passes, label, True)

            def callback(_pipeline, step, _timestep, _tensors):
                report(0.15 + 0.55 * (index + (step + 1) / STEPS) / passes, f"{label}, step {step + 1}/{STEPS}")
                return {}

            return callback

        options = dict(prompt="[MVFLUX]", prompt_embeds=None, pooled_prompt_embeds=None, height=VIEW_SIZE,
                       width=6 * VIEW_SIZE, n_rows=1, n_cols=6, num_inference_steps=STEPS,
                       guidance_scale=GUIDANCE_SCALE, max_sequence_length=512, generator=generator)
        self.pipeline.set_adapters(adapter_names=ADAPTERS, adapter_weights=[1.0, 0.0])
        strips = {"lit": self.pipeline(control_image=_control(rendered["normal"], rendered["ccm"]),
                                       dual_image=condition, callback_on_step_end=sampling(0, "sampling the views"),
                                       **options).images[0]}
        if params["delight"]:
            self.pipeline.set_adapters(adapter_names=ADAPTERS, adapter_weights=[0.0, 1.0])
            strips["delit"] = self.pipeline(control_image=strips["lit"],
                                             callback_on_step_end=sampling(1, "removing the lighting"),
                                             **options).images[0]

        torch.manual_seed(params["seed"])
        for index, (name, strip) in enumerate(strips.items()):
            report(0.7 + 0.29 * index / passes, f"upscaling the {name} views", True)
            upscaled = self.upscaler(_grid(strip))
            side = upscaled.width // 3
            for number in range(6):
                row, column = divmod(number, 3)
                view = upscaled.crop((column * side, row * side, (column + 1) * side, (row + 1) * side))
                silhouette = rendered["alpha"].crop((column * VIEW_SIZE, row * VIEW_SIZE, (column + 1) * VIEW_SIZE,
                                                     (row + 1) * VIEW_SIZE)).resize((side, side),
                                                                                      Image.Resampling.BILINEAR)
                view.putalpha(silhouette)
                buffer = io.BytesIO()
                view.save(buffer, "PNG")
                files[f"{name}_view_{number}.png"] = buffer.getvalue()
        torch.cuda.empty_cache()
        return {**files, "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "seconds": round(time.monotonic() - started, 3)}
