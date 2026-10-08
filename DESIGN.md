# AssetFactory — Design

AssetFactory makes game assets — images, meshes, textures, sound effects, music and speech — for AI agents and people,
through a REST API and a web UI.

1. **`shared/`** — modules of code shared by two or more stages, never by jobs; without classes; one module per
   subject.
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
   - Order: arrival. Calls to the loaded `Worker` go first while the oldest other call has waited
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
   - Jobs and batches: params, status, outputs, work time and cloud cost. Work time is the time of stage
     execution; queue waits are not counted.
8. **`Job`** — the base class of a job.
   - A subclass declares `Params`: a pydantic class with the names, types, limits and enums of the params, without
     default values. `Params` also declares the file inputs, as lists of file_ids with their accepted extensions and count.
   - A subclass implements `async run(ctx)`: a linear sequence of stage calls: a condition on the params may skip a
     stage, but never puts one stage in place of another. After the last stage, `run` assembles the job's outputs
     from the stage outputs (for example, writes the textured GLB).
   - Through `ctx` a subclass:
     - calls stages: `await ctx.local(stage, …)` goes to `LocalQueue`, `await ctx.cloud(app, …)` to `CloudQueue`;
     - reports progress;
     - gets its folder.
   - A cloud stage is always the first stage of a job.
   - Next to each subclass lies a yaml file: the job description (at most one paragraph), its modality (image,
     3d, audio, music or speech) and one line per param and output.
   - `Params` and the yaml give a JSON schema; it validates requests and builds the guide and the UI forms.
9. **`Job` subclasses**, one file in `jobs/` per line. Identical code in different jobs is duplicated. Every job whose
   chain has `decimate` has a required `target_faces`.
   1. `zimage_text_to_image.py`
   2. `zimage_image_to_image.py`
   3. `flux2_text_to_image.py`
   4. `flux2_image_edit.py`
   5. `triposg_shape.py` — `bgremove` → triposg → `clean` → `decimate`
   6. `hy2_shape.py` — `bgremove` → hunyuan3d-2 → `clean` → `decimate`
   7. `hy2_textured.py` — `bgremove` → hunyuan3d-2 → `clean` → `decimate` → `unwrap` → Hunyuan3D-Paint
   8. `hy2mv_shape.py` — `bgremove` → hunyuan3d-2mv → `clean` → `decimate`
   9. `hy2mv_textured.py` — `bgremove` → hunyuan3d-2mv → `clean` → `decimate` → `unwrap` → Hunyuan3D-Paint
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
    - `seed` is a param of the jobs whose stages are seeded: an integer or `"random"`; the Api replaces `"random"`
      with a random integer when it creates the job.
    - `count` in a request gives that many runs of `run`, with seed, seed+1 and so on.
11. **`Notifier`** — the `notify_url` webhook.
    - It fires when a job or a batch ends.
    - The text holds the status, the error, the `file_id` and path of each output, and "work took N s (X $)".
    - It retries delivery while the factory runs.
12. **`Api`** (FastAPI):
    - `GET /api/usage` — the guide from `usage.md` and every schema in one response;
    - `GET /api/health`;
    - `GET /api/jobs/schema` — every job's description, modality, params JSON schema and outputs as JSON (the data
      `/api/usage` renders);
    - `POST /api/files`, `GET /api/files?origin=`, `GET /api/files/{id}`, `GET /api/files/{id}/content`;
    - `POST /api/jobs {job, params, count, notify_url}`, every field explicit, `notify_url` may be null;
    - `POST /api/batches {jobs, notify_url}` — all jobs are created or none; `GET /api/batches/{id}?wait=`;
    - `GET /api/jobs?status=&job=&batch_id=` (history, paged; `job` takes several names) and
      `GET /api/jobs/{id}?wait=`;
    - cancel a job, retry it: a failed or cancelled job with the same seed, a succeeded one with a new seed; delete
      it with its files;
    - `GET /api/system` — the loaded `Worker`, both queues, RAM, VRAM, CPU and NVIDIA GPU load, the Modal credits
      used and billed of the month; `POST /api/system/unload`;
    - `GET /api/events` — SSE.
13. **`Ui`** — one web page over `Api`:
    - header: load bars of RAM, VRAM, CPU and NVIDIA GPU, and the Modal credits used and billed of the month;
    - left: the queue of the jobs not started yet, each with a cancel button; right: the history with filters by
      model, task and status, live updates, result viewing (GLB in model-viewer, images, audio), retry, details,
      delete;
    - a "New" modal: model and task choice with a filter by the model's modality, the form from the schema with the
      recommended values of `ui/models.json`, file upload.
14. **CLI** — installs environments and weights. A job whose environment or weights are absent is refused with the
    install command.
15. **Shutdown** (Ctrl+C): cloud calls are cancelled with their containers terminated, the idle containers of the Modal
    apps are stopped, every `Worker` is stopped. At start, jobs left unfinished in `Store` are marked failed with
    "factory stopped".
