"""Modal app of the trellis2 cloud stage: the raw mesh of one object from an image with TRELLIS.2, as `raw.glb`. In one
container: background removal with the `bgremove` stage code (BiRefNet), the sparse structure and shape latents (no
texture latent), the shape decoder, dual-contouring remesh and simplification with CuMesh on the GPU. The mesh has no
UVs and no attributes.

The weights are in the Modal Volume `VOLUME`, mounted at WEIGHTS: the Hugging Face cache of the repositories, the
converted weight files of the shape models in CONVERTED (each tensor in the dtype of the model's parameter or buffer of
that name) and the file INDEX that `download_weights` writes. The models are built without weights and their tensors are
loaded straight onto the GPU. The Triton kernel cache and the FlexGEMM autotuning cache are kept in the Volume and
committed after every call. Calls: see the package docstring."""
import time
from pathlib import Path

import modal
from cloud import load_strict, read_image, run_call, silent

APP_NAME = "assetfactory-trellis2"
VOLUME = "assetfactory-trellis2-weights"
WEIGHTS_APP_NAME = f"{APP_NAME}-weights"
CLASS = "Trellis2"
# GPU type, CPU cores, memory in MiB, seconds a container stays up without a call, and seconds a call or a container
# start may take. The last four are guessed.
GPU = "L40S"
CPU = 2
MEMORY_MIB = 32768
SCALEDOWN_SECONDS = 60
TIMEOUT_SECONDS = 900

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = "/weights"
SOURCE = "/opt/trellis2"
SHARED = "/opt/af"
TRELLIS_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CUMESH_COMMIT = "12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_COMMIT = "6dd94a859c26ee8246888502eada3dd8ad85532e"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
NVDIFFREC_COMMIT = "b296927cc7fd01c2ac1087c8065c4d7248f72da4"
UTILS3D_COMMIT = "9a4eb15e4021b67b12c460c7057d642626897ec8"
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
                    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")

PIPELINE_REPO = "microsoft/TRELLIS.2-4B"
# Written by download_weights, read by load: the `args` of the repository's pipeline.json and, for each model in
# SHAPE_MODELS, the paths of its config file and its converted weights file.
INDEX = f"{WEIGHTS}/trellis2.json"
CONVERTED = Path(WEIGHTS) / "trellis2"
TRITON_CACHE = f"{WEIGHTS}/triton"
FLEXGEMM_CACHE = f"{WEIGHTS}/flex_gemm/autotune_cache.json"
# The models of the sparse structure and of the shape.
SHAPE_MODELS = ["sparse_structure_flow_model", "sparse_structure_decoder", "shape_slat_flow_model_512",
                "shape_slat_flow_model_1024", "shape_slat_decoder"]
# Edge of the sparse structure grid. Documented.
STRUCTURE_RESOLUTION = 32
# Band in voxels of the dual-contouring remesh, and whether its vertices are projected back to the decoded surface.
# Documented.
REMESH_BAND = 1
REMESH_PROJECT = 0
# Largest hole perimeter that the hole filling closes, in the units of the unit cube. Documented.
HOLE_PERIMETER = 3e-2

def pinned(url, commit, path):
    return (f"git clone -q {url} {path} && git -C {path} checkout -q {commit} && "
            f"git -C {path} submodule update -q --init --recursive")


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_ATTN_WHEEL, "transformers==4.57.1", "huggingface_hub==0.36.0", "safetensors==0.6.2",
                 "accelerate==1.12.0", "numpy==1.26.4", "pillow==11.3.0", "opencv-python-headless==4.10.0.84",
                 "imageio==2.37.0", "trimesh==4.8.3", "easydict==1.13", "tqdm==4.67.1", "zstandard==0.25.0",
                 "plyfile==1.1.2", "kornia==0.8.1", "timm==1.0.20", "einops==0.8.0", "ninja==1.13.0",
                 "setuptools==75.8.0", "wheel==0.45.1",
                 f"utils3d @ git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_COMMIT}")
    # The CUDA extensions are compiled with gcc for compute capability 8.9 (L40S).
    .env({"TORCH_CUDA_ARCH_LIST": "8.9", "CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -pthread -shared",
          "LDCXXSHARED": "g++ -pthread -shared", "HF_HOME": f"{WEIGHTS}/hf", "PYTHONPATH": f"{SOURCE}:{SHARED}"})
    .run_commands(
        pinned("https://github.com/NVlabs/nvdiffrast.git", NVDIFFRAST_COMMIT, "/opt/ext/nvdiffrast"),
        "pip install --no-build-isolation /opt/ext/nvdiffrast",
        pinned("https://github.com/JeffreyXiang/nvdiffrec.git", NVDIFFREC_COMMIT, "/opt/ext/nvdiffrec"),
        "pip install --no-build-isolation /opt/ext/nvdiffrec",
        pinned("https://github.com/JeffreyXiang/CuMesh.git", CUMESH_COMMIT, "/opt/ext/CuMesh"),
        "pip install --no-build-isolation /opt/ext/CuMesh",
        pinned("https://github.com/JeffreyXiang/FlexGEMM.git", FLEXGEMM_COMMIT, "/opt/ext/FlexGEMM"),
        "pip install --no-build-isolation /opt/ext/FlexGEMM",
        pinned("https://github.com/microsoft/TRELLIS.2.git", TRELLIS_COMMIT, SOURCE),
        f"pip install --no-build-isolation --no-deps {SOURCE}/o-voxel",
    )
    .add_local_file(ROOT / "stages" / "triposg" / "bgremove.py", f"{SHARED}/bgremove.py")
    .add_local_file(ROOT / "shared" / "mapped.py", f"{SHARED}/mapped.py")
    .add_local_file(ROOT / "worker" / "context.py", f"{SHARED}/context.py")
)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)
weights_app = modal.App(WEIGHTS_APP_NAME, image=image)


@weights_app.function(volumes={WEIGHTS: volume}, timeout=3600)
def download_weights(hf_token):
    """Fetches the files that `load` reads into the Volume, writes the converted weight files and INDEX; files already
    written are kept. The converted file of a model holds the tensors of its weights file that have a parameter or
    buffer of that name in the model, in the dtype of that parameter or buffer."""
    import json
    import os

    import bgremove
    import mapped
    from accelerate import init_empty_weights
    from huggingface_hub import file_exists, hf_hub_download, snapshot_download
    from safetensors import safe_open
    from trellis2 import models as trellis_models

    os.environ["HF_TOKEN"] = hf_token
    args = json.loads(Path(hf_hub_download(PIPELINE_REPO, "pipeline.json")).read_text(encoding="utf-8"))["args"]
    CONVERTED.mkdir(parents=True, exist_ok=True)
    index = {}
    for name in SHAPE_MODELS:
        entry = args["models"][name]
        if file_exists(PIPELINE_REPO, f"{entry}.json"):
            repo, stem = PIPELINE_REPO, entry
        else:
            repo, stem = "/".join(entry.split("/")[:2]), "/".join(entry.split("/")[2:])
        config_file = hf_hub_download(repo, f"{stem}.json")
        weights_file = hf_hub_download(repo, f"{stem}.safetensors")
        converted = CONVERTED / f"{name}.safetensors"
        if not converted.exists():
            config = json.loads(Path(config_file).read_text(encoding="utf-8"))
            with init_empty_weights():
                state = getattr(trellis_models, config["name"])(**config["args"]).state_dict()
            with safe_open(weights_file, "pt") as file:
                mapped.save_weights(converted, {key: file.get_tensor(key).to(state[key].dtype)
                                                for key in file.keys() if key in state})
        index[name] = [config_file, str(converted)]
    snapshot_download(args["image_cond_model"]["args"]["model_name"], allow_patterns=["*.json", "*.safetensors"])
    bgremove.download()
    Path(INDEX).write_text(json.dumps({"args": args, "models": index}), encoding="utf-8")
    volume.commit()


@app.cls(gpu=GPU, cpu=CPU, memory=MEMORY_MIB, max_containers=1, volumes={WEIGHTS: volume}, timeout=TIMEOUT_SECONDS,
         scaledown_window=SCALEDOWN_SECONDS, enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True},
         env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TRITON_CACHE_DIR": TRITON_CACHE,
              "FLEX_GEMM_AUTOTUNE_CACHE_PATH": FLEXGEMM_CACHE})
class Trellis2:
    @modal.enter(snap=True)
    def load(self):
        """Loads every model onto the GPU, then runs one short generation. The BiRefNet tensors are copied to the GPU
        and their memory maps released."""
        import json

        import bgremove
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from torchvision import transforms
        from transformers import DINOv3ViTModel
        from trellis2 import models
        from trellis2.modules import image_feature_extractor
        from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers

        Path(FLEXGEMM_CACHE).parent.mkdir(parents=True, exist_ok=True)
        index = json.loads(Path(INDEX).read_text(encoding="utf-8"))
        args = index["args"]
        loaded = {}
        for name, (config_file, weights_file) in index["models"].items():
            config = json.loads(Path(config_file).read_text(encoding="utf-8"))
            with init_empty_weights():
                model = getattr(models, config["name"])(**config["args"])
            load_strict(model, load_file(weights_file, device="cuda"))
            loaded[name] = model.to("cuda")

        extractor_class = image_feature_extractor.DinoV3FeatureExtractor
        extractor = extractor_class.__new__(extractor_class)
        extractor.model_name = args["image_cond_model"]["args"]["model_name"]
        extractor.model = DINOv3ViTModel.from_pretrained(extractor.model_name, device_map="cuda").eval()
        extractor.image_size = 512
        extractor.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        structure, shape = args["sparse_structure_sampler"], args["shape_slat_sampler"]
        self.pipeline = Trellis2ImageTo3DPipeline(
            models=loaded,
            sparse_structure_sampler=getattr(samplers, structure["name"])(**structure["args"]),
            shape_slat_sampler=getattr(samplers, shape["name"])(**shape["args"]),
            sparse_structure_sampler_params=structure["params"],
            shape_slat_sampler_params=shape["params"],
            shape_slat_normalization=args["shape_slat_normalization"],
            image_cond_model=extractor,
            rembg_model=None,
            low_vram=False,
        )
        self.pipeline.cuda()

        self.remover = bgremove.load_on_gpu()
        example = Path(SOURCE) / "assets" / "example_image" / "T.png"
        self._generate(example.read_bytes(), {"resolution": 1024, "steps": 2, "decimation_target": 100000, "seed": 0},
                       silent)

    @modal.method()
    def run(self, image, params, progress):
        return run_call(self._generate, progress, image, params, last=volume.commit)

    def _generate(self, data, params, report):
        import bgremove
        import cumesh
        import numpy as np
        import torch
        import trimesh
        from trellis2.pipelines.samplers import flow_euler

        started = time.monotonic()
        source = read_image(data)
        report(0.0, "removing the background", True)
        rgba = bgremove.remove_background(self.remover, source)

        total = params["steps"] * (2 if params["resolution"] == 512 else 3)
        done = [0]

        def counted(iterable, desc, disable):
            for item in iterable:
                yield item
                done[0] += 1
                report(0.05 + 0.75 * done[0] / total, f"sampling step {done[0]}/{total}")

        flow_euler.tqdm = counted
        report(0.05, "encoding the image", True)
        pipeline = self.pipeline
        sampler = {"steps": params["steps"]}
        with torch.no_grad():
            condition = pipeline.preprocess_image(rgba)
            torch.manual_seed(params["seed"])
            cond_512 = pipeline.get_cond([condition], 512)
            coords = pipeline.sample_sparse_structure(cond_512, STRUCTURE_RESOLUTION, 1, sampler)
            if params["resolution"] == 512:
                shape_slat = pipeline.sample_shape_slat(cond_512, pipeline.models["shape_slat_flow_model_512"], coords,
                                                        sampler)
                resolution = 512
            else:
                shape_slat, resolution = pipeline.sample_shape_slat_cascade(
                    cond_512, pipeline.get_cond([condition], 1024), pipeline.models["shape_slat_flow_model_512"],
                    pipeline.models["shape_slat_flow_model_1024"], 512, params["resolution"], coords, sampler)
            torch.cuda.empty_cache()
            report(0.8, "decoding the shape", True)
            meshes, _ = pipeline.decode_shape_slat(shape_slat, resolution)

        report(0.85, "remeshing", True)
        mesh = cumesh.CuMesh()
        mesh.init(meshes[0].vertices, meshes[0].faces)
        mesh.fill_holes(max_hole_perimeter=HOLE_PERIMETER)
        vertices, faces = mesh.read()
        bvh = cumesh.cuBVH(vertices, faces)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces, center=torch.zeros(3, device="cuda"), scale=(resolution + 3 * REMESH_BAND) / resolution,
            resolution=resolution, band=REMESH_BAND, project_back=REMESH_PROJECT, bvh=bvh))
        report(0.93, "simplifying", True)
        mesh.simplify(params["decimation_target"])
        vertices, faces = mesh.read()
        report(0.97, "writing", True)
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        # The decoded Z-up volume to Y-up.
        glb = trimesh.Trimesh(vertices=np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1),
                              faces=faces, process=False).export(file_type="glb", include_normals=False)
        torch.cuda.empty_cache()
        return {"raw.glb": glb, "resolution": int(resolution), "vertices": len(vertices), "faces": len(faces),
                "seconds": round(time.monotonic() - started, 3)}
