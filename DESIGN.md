# AssetFactory — Design

AssetFactory makes game assets — images, meshes, textures, sound effects, music and speech — for AI agents and people,
through a REST API and a web UI.

1. **`shared/`** — modules of code shared by two or more stages, without classes: `meshops` (GLB read and write,
   vertex welding, welded normals, UV overlap count), `diffusers_common` (fitting a text encoder, a transformer and a
   VAE onto 8 GB, the sampling-step callback).
2. **`Worker`** — a process in the venv of one of the factory's environments; it talks to the core with JSON over a
   pipe. Loaded models stay in the process until it is unloaded. It runs offline.
3. **Local stage** — a function a `Worker` runs; its inputs and outputs are files. Local stages:
   - `bgremove` (BiRefNet), z-image, flux2;
   - triposg, hunyuan3d-2, hunyuan3d-2mv, sf3d, Hunyuan3D-Paint, MV-Adapter views;
   - `clean`, `decimate`, `unwrap`, `bake` (projection of views into the UV texture), `fill` (FlexPainter completion
     of empty texels);
   - stable-audio, ace-step, chatterbox, qwen-tts.
4. **Cloud stage** — a function in a Modal app: trellis2, hunyuan3d-2.1, UniTEX views. Background removal runs in the
   same container, with the `bgremove` code copied into it.
5. **`LocalQueue`** — one queue for the whole machine.
   - It runs one local stage at a time.
   - Order: call priority, then arrival. Calls to the loaded `Worker` go first while the oldest other call has waited
     less than a limit.
   - Before switching to another `Worker` it checks free RAM, computed at that moment.
   - An idle `Worker` is unloaded after a timeout and on command.
6. **`CloudQueue`** — one queue per Modal app.
   - A container runs one call at a time.
   - The queue watches the container and sends the next call the moment the container is free.
   - Results are downloaded asynchronously; sending the next call does not wait for a download.
   - The queues of different apps run in parallel with each other and with `LocalQueue`.
   - It counts cloud seconds and dollars.
7. **`Store`** — SQLite.
   - Files: every input and output is stored by the factory and has a `file_id`.
   - Jobs and batches: params, inputs, status, outputs, work time and cloud cost. Work time is the time of stage
     execution; queue waits are not counted.
8. **`Job`** — the base class of a job.
   - A subclass declares `Params`: a pydantic class with the names, types, limits and enums of the params, without
     default values.
   - A subclass implements `async run(ctx)`: a strictly linear sequence of stage calls, without branches.
   - Through `ctx` a subclass:
     - calls stages: `await ctx.local(stage, …)` goes to `LocalQueue`, `await ctx.cloud(app, …)` to `CloudQueue`;
     - reports progress;
     - gets its folder.
   - A cloud stage is always the first stage of a job.
   - Next to each subclass lies a yaml file: the job description (at most one paragraph) and one line per param,
     input and output.
   - `Params` and the yaml give a JSON schema; it validates requests and builds the guide and the UI forms.
9. **`Job` subclasses**, one file in `jobs/` per line. Identical code in different jobs is duplicated. Every job that
   makes a shape has a required `target_faces`.
   1. `zimage_text_to_image.py`
   2. `zimage_image_to_image.py`
   3. `flux2_text_to_image.py`
   4. `flux2_image_edit.py`
   5. `triposg_shape.py` — `bgremove` → triposg → `clean` → `decimate`
   6. `hy2_shape.py` — `bgremove` → hunyuan3d-2 → `clean` → `decimate`
   7. `hy2_textured.py` — `bgremove` → hunyuan3d-2 → `clean` → `decimate` → Hunyuan3D-Paint
   8. `hy2mv_shape.py` — `bgremove` → hunyuan3d-2mv → `clean` → `decimate`
   9. `hy2mv_textured.py` — `bgremove` → hunyuan3d-2mv → `clean` → `decimate` → Hunyuan3D-Paint
   10. `trellis2_shape.py` — trellis2 (cloud) → `clean` → `decimate`
   11. `hy21_shape.py` — hunyuan3d-2.1 (cloud) → `clean` → `decimate`
   12. `sf3d_textured.py` — `bgremove` → sf3d
   13. `mesh_unwrap.py` — `unwrap`
   14. `hypaint_texture.py` — `bgremove` → Hunyuan3D-Paint
   15. `mvadapter_texture.py` — `bgremove` → MV-Adapter views → `bake` → `fill`
   16. `unitex_texture.py` — UniTEX views (cloud) → `bake` → `fill`
   17. `sfx_text.py`
   18. `sfx_audio_to_audio.py`
   19. `sfx_inpaint.py`
   20. `sfx_continue.py`
   21. `music_text.py`
   22. `music_repaint.py`
   23. `music_extend.py`
   24. `music_cover.py`
   25. `chatterbox_speech.py`
   26. `qwen_voicedesign_speech.py`
10. **`Runner`** — runs every job as an asyncio task and keeps its status, cancel and failure in `Store`.
    - `count` in a request gives that many runs of `run`, with seed, seed+1 and so on.
    - The job's priority is the priority of its calls.
11. **`Notifier`** — the `notify_url` webhook.
    - It fires when a job or a batch ends.
    - The text holds the status, the error, the `file_id` and path of each output, and "work took N s (X $)".
    - It retries delivery while the factory runs.
12. **`Api`** (FastAPI):
    - `GET /api/usage` — the guide from `usage.md` and every schema in one response;
    - `GET /api/health`;
    - `POST /api/files`, `GET /api/files?kind=&origin=`, `GET /api/files/{id}`, `GET /api/files/{id}/content`;
    - `POST /api/jobs {job, params, inputs, count?, seed?, priority?, label?, notify_url?}`;
    - `POST /api/batches` — all jobs are created or none; `GET /api/batches/{id}?wait=`;
    - `GET /api/jobs?status=&job=&batch_id=` (history, paged) and `GET /api/jobs/{id}?wait=`;
    - cancel a job, retry it with the same or a new seed, delete it with its files;
    - `GET /api/system` — the loaded `Worker`, both queues, RAM and VRAM; `POST /api/system/unload`;
    - `GET /api/events` — SSE.
13. **`Ui`** — a web page over `Api`:
    - forms from the schemas;
    - file upload;
    - queue and history;
    - result viewing: GLB in model-viewer, images, audio.
14. **CLI** — installs environments and weights; runs a smoke test of a job on reference inputs. A job whose
    environment or weights are absent is refused with the install command.
15. **Shutdown** (Ctrl+C): cloud calls are cancelled, Modal apps are stopped, every `Worker` is stopped. At start,
    jobs left unfinished in `Store` are marked failed with "factory stopped".
