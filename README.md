# AssetFactory

AssetFactory makes game assets with AI models: images, 3D meshes from images, textures for meshes, sound effects,
music and speech. Most models run on your own GPU; three run on cloud GPUs at [Modal](https://modal.com). You use it
from a web page in your browser; programs and AI agents use the same factory through a REST API. Everything you make
stays in a history with its settings and files.

---

## Install

You need:

* **Windows 11**
* **an NVIDIA GPU with 8 GB VRAM** (RTX 40 series) and a driver that supports CUDA 13.0
* **32 GB RAM**
* **about 170 GB of free disk** for all local models: about 150 GB of weights and 16 GB of Python
  environments, plus room for your outputs
* **PowerShell 7** (`pwsh`), **uv** and **git** on your PATH
* for the environments `triposg`, `hunyuan3d`, `sf3d` and `flexpainter`: **CUDA toolkit 12.6** and **Visual Studio
  2022** with "Desktop development with C++" (they compile GPU code during install)

### 1. Get the code

```
git clone https://github.com/S-Elisei/AssetFactory.git
cd AssetFactory
```

### 2. Create the factory's own environment

```
pwsh envs_spec\install.ps1 core
```

### 3. Install the models

Each group of models has its own Python environment in `envs\` and its weights in `models\`. Install everything,
or one environment or cloud app at a time:

```
envs\core\Scripts\python.exe -m core.install all
envs\core\Scripts\python.exe -m core.install <name>
```

The local environments are `acestep`, `chatterbox`, `diffusers`, `flexpainter`, `hunyuan3d`, `meshops`, `qwentts`,
`sf3d`, `stableaudio3` and `triposg`; the cloud apps are `hunyuan3d21`, `trellis2` and `unitex`. A job whose
environment, weights or cloud app is missing is refused with the install command it needs.

### 4. A Hugging Face token

Some weights are served only to a signed-in Hugging Face account that has accepted their license or been granted
access: [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d),
[stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) and, for the
cloud app `trellis2`, [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

1. Sign in at huggingface.co, open these pages and accept the license or request access.
2. Create a token with read access (Settings → Access Tokens).
3. Save the token as the only line of the file `models\hf\token`. The factory keeps its Hugging Face data in
   `models\hf`, so a login stored elsewhere on your computer is not used.

### 5. A Modal account for the cloud models

`hy21_shape` (Hunyuan3D-2.1), `trellis2_shape` (TRELLIS.2) and `unitex_texture` (UniTEX) run on Modal GPUs, billed
by Modal per second of container time. Their job descriptions state the cloud work a job requires.

1. Create a Modal account and add a payment method.
2. Store a Modal token: `envs\core\Scripts\modal.exe token new`. The token is saved in `.modal.toml` in your user
   folder.
3. Install the cloud apps (`core.install hunyuan3d21`, `trellis2`, `unitex`, or `all`). The install downloads the
   weights into a Modal volume, deploys the app and starts one container, which takes the memory snapshot that later
   containers start from.

### Where things go

Environments are in `envs\`, weights in `models\`, the package cache in `.cache\`. Your jobs, files and the database
are in `data\`. Settings are in `config.yaml`.

---

## Run

```
envs\core\Scripts\python.exe -m core
```

Then open **http://127.0.0.1:8700/**. The guide for programs and AI agents is at
**http://127.0.0.1:8700/api/usage**.

The factory listens only on this computer and has no login. **Ctrl+C** stops it: running cloud calls are cancelled,
the idle cloud containers are stopped and every model process is stopped. Jobs left unfinished are marked failed
with "factory stopped" at the next start.

### The web page

* **Header:** load of RAM, VRAM, CPU and GPU, and the Modal credits used and billed this month.
* **Queue:** the jobs not started yet, each with a cancel button.
* **History:** every job with its status and outputs; meshes open in a 3D viewer, images and audio play in place.
  A job can be retried or deleted with its files. Filters by model, task and status.
* **New:** pick a model and a task, fill the form and send. The form starts with recommended values.

---

## What the factory does

**Twenty-six jobs.** A job is a recipe of stages with its own params:

| modality | job | what it does |
|---|---|---|
| image | `zimage_text_to_image` | text to image with Z-Image-Turbo |
| image | `zimage_image_to_image` | re-renders an image following a prompt with Z-Image-Turbo |
| image | `flux2_text_to_image` | text to image with FLUX.2 [klein] 4B |
| image | `flux2_image_edit` | edits or combines reference images following an instruction with FLUX.2 [klein] 4B |
| 3d | `triposg_shape` | one image to an untextured mesh with TripoSG |
| 3d | `hy2_shape` | one image to an untextured mesh with Hunyuan3D-2 |
| 3d | `hy2_textured` | one image to a textured mesh with Hunyuan3D-2 and Hunyuan3D-Paint |
| 3d | `hy2mv_shape` | front view plus up to three more views to an untextured mesh with Hunyuan3D-2 multiview |
| 3d | `hy2mv_textured` | the same, textured with Hunyuan3D-Paint |
| 3d | `hy21_shape` | one image to an untextured mesh with Hunyuan3D-2.1, on a Modal GPU |
| 3d | `trellis2_shape` | one image to an untextured mesh with TRELLIS.2, on a Modal GPU |
| 3d | `sf3d_textured` | one image to a textured, UV-unwrapped mesh with stable-fast-3d |
| 3d | `mesh_unwrap` | UV-unwraps a mesh into charts packed without overlapping texels |
| 3d | `hypaint_texture` | textures a UV-unwrapped mesh from one reference image with Hunyuan3D-Paint |
| 3d | `mvadapter_texture` | textures a UV-unwrapped mesh from one reference image with MV-Adapter views |
| 3d | `unitex_texture` | textures a UV-unwrapped mesh from one reference image with UniTEX views, on a Modal GPU; lit and delit textures |
| audio | `sfx_text` | text to sound effect with Stable Audio 3 Small-SFX |
| audio | `sfx_audio_to_audio` | a new version of a sound clip steered by a prompt |
| audio | `sfx_inpaint` | regenerates a time range of a sound clip |
| audio | `sfx_continue` | continues a sound clip |
| music | `music_text` | text and optional lyrics to music with ACE-Step 1.5 |
| music | `music_repaint` | regenerates a time range of a track |
| music | `music_extend` | continues a track |
| music | `music_cover` | re-renders a track in the style of a caption |
| speech | `chatterbox_speech` | text to speech with Chatterbox Multilingual V3, with voice cloning from a recording |
| speech | `qwen_voicedesign_speech` | text to speech with Qwen3-TTS 1.7B VoiceDesign in a voice described in words |

**Meshes.** The shape jobs other than `sf3d_textured` clean the generated mesh into one closed manifold surface and
decimate it to the requested face count. Meshes are GLB, Y up, with the front facing +Z. `hypaint_texture`,
`mvadapter_texture` and `unitex_texture` keep the geometry and the UVs of the input mesh; they need UVs without
overlapping charts, which `mesh_unwrap` makes. `mvadapter_texture` and `unitex_texture` project their views into the UV
layout and complete the texels no view sees with FlexPainter.

**You choose the job.** Every request names its job; the factory never picks or swaps a model.

**Every setting is sent.** Params have no default values: a request lists every param of its job. `notify_url: null`
is a valid value. The web form starts with recommended values.

**One local stage at a time.** Local stages run one at a time on the GPU, in arrival order; the loaded model serves
the next stage that needs it, and an idle model is unloaded after a timeout. Each cloud app has its own queue, which
runs in parallel with the local one.

**History.** Every job keeps its params, seeds, status, work time, cloud cost and output files. Every input and output
file has a `file_id`; deleting a job deletes its files.

**Chaining.** Any output can be the input of another job by its `file_id`: an image into a shape job, the mesh into
`mesh_unwrap`, the unwrapped mesh into a texture job.

**Batches.** Several jobs can be sent together; either all are accepted or none.

**Notifications.** A job or batch can carry a `notify_url`; the factory posts a short text there when the work ends,
with the status, the error, the output files and the work time and cloud cost.

**A guide for AI agents.** `/api/usage` explains the whole API and lists every job with its description, params and
outputs.

**Settings.** `config.yaml`: `cloud_scaledown_seconds`, the seconds a Modal container stays up after its last call.

**Licenses.** Every model is under its own license, as its authors publish it; the licenses differ, also in what they
allow for commercial use.

---

MIT licensed. See [LICENSE](LICENSE). The models have their own licenses.
