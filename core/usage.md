# AssetFactory - guide for agents

AssetFactory makes game assets on this machine: images, meshes (GLB), textures, sound effects, music and speech. A
request names a job and gives its params and input files; the factory queues the job, runs its stages on the local GPU
or on Modal cloud GPUs, and stores the output files. The base URL is `{{BASE}}`. Requests and responses are JSON,
except file uploads (multipart) and file downloads.

The commands below are written for `curl.exe`. A JSON request body is sent from a file with
`--data-binary "@file.json"`.

## Jobs

A job is a named recipe: `zimage_text_to_image`, `hy2_textured`, `music_text` and so on. The catalog at the end of this
guide lists every job with its description, its params as a JSON schema, its inputs and its outputs.
- `GET {{BASE}}/api/jobs/schema` answers the catalog as JSON: `{job name: {description, params, inputs, outputs}}`.
- Every param of a job is required; params have no default values. A param that is not in the schema is refused.
- `seed` is a param of the jobs whose stages are seeded: an integer from 0 to {{SEED_MAX}}, or `"random"`. The factory
  replaces `"random"` with a random integer when it creates the job; the job object holds that integer.
- `count` runs the job that many times at the same time, as items. Item `i` uses the seed plus `i`. Each item has its
  own output files.
- A job is refused when an environment, the weights or the Modal app of one of its stages is not installed. The refusal
  names the command that installs it; the command runs from the repository root.

## Submitting a job

`POST {{BASE}}/api/jobs` with a JSON body; answer 201 with the job object.

| field | required | meaning |
|---|---|---|
| `job` | yes | job name from the catalog |
| `params` | yes | `{param name: value}` for every param of the job |
| `inputs` | per job | `{input name: [file_id, ...]}` |
| `count` | no, 1 | number of items, at least 1 |
| `priority` | no, 0 | integer; a higher priority is served first |
| `label` | no | free text, stored with the job |
| `notify_url` | no | http or https URL that receives a notification when the job ends |

```
curl.exe -s -X POST {{BASE}}/api/jobs -H "Content-Type: application/json" --data-binary "@job.json"
```
with `job.json`
```
{"job": "zimage_text_to_image",
 "params": {"prompt": "a wooden treasure chest, game asset, white background", "width": 1024, "height": 1024,
            "steps": 8, "seed": "random"},
 "count": 2}
```

## Files and chaining

Every input and every output is a file with a `file_id`.
- Upload: `curl.exe -s -F "file=@concept.png" {{BASE}}/api/files` answers 201 with the file object. Accepted extensions
  by kind: {{UPLOADS}}.
- File object: `file_id`, `kind` (`image`, `mesh` or `audio`), `origin` (`upload` or `output`), `name`, `path`
  (absolute path on this machine), `size`, `created`, `job_id` and `item` (both null for an upload).
- `GET {{BASE}}/api/files?kind=&origin=` lists files, newest first. `GET {{BASE}}/api/files/{file_id}` answers one file
  object. `GET {{BASE}}/api/files/{file_id}/content` downloads the file.
- An input takes the `file_id` of an upload or of any job output of the kind the input declares. An output of one
  job becomes an input of another by passing its `file_id`:

```
{"job": "trellis2_shape",
 "params": {"resolution": 1024, "steps": 12, "target_faces": 20000, "seed": "random"},
 "inputs": {"image": ["<file_id of an output of zimage_text_to_image>"]}}
```

## Waiting for a job

`GET {{BASE}}/api/jobs/{job_id}?wait=SECONDS` answers when the job has ended or after SECONDS (0 to {{WAIT_MAX}}),
whichever comes first, with the job object. A job has ended when its `status` is `succeeded`, `failed` or `cancelled`.

```
curl.exe -s "{{BASE}}/api/jobs/j_0123456789ab?wait=60"
```

## Job object

- `job_id`, `job` (job name), `params` (with the integer seed), `inputs`, `count`, `priority`, `label`, `notify_url`,
  `batch_id`, `seq` (position in the history).
- `status`: `queued` (submitted, no stage has reported yet), `running`, `succeeded`, `failed`, `cancelled`.
- `error`: null, or `{"kind", "message"}`. Kinds: `input` (a stage refused an input file; the message says why),
  `oom` (a stage ran out of memory), `failed` (a stage raised an error; the message quotes it), `stopped` (the factory
  stopped while the job was queued or running).
- `outputs`: one list of `file_id`s per item. A failed or cancelled job keeps the outputs registered before it ended.
- `created`, `started`, `finished`: epoch seconds, null until they happen.
- `work_seconds`, `cloud_dollars`: the time of stage execution, queue waits not counted, and the cloud cost.
- `progress`: while the job has not ended, `{"fraction", "message"}` (fraction 0 to 1 of the whole job, the message
  names the stage and its state); null after the job has ended.

The first item that fails ends the job as `failed` and cancels the other items.

## Batches

`POST {{BASE}}/api/batches` with `{"jobs": [<job request>, ...], "label": ..., "notify_url": ...}` (at least one job;
`label` and `notify_url` are optional). Every job is checked first; all jobs are created or none. The answer (201) holds `batch_id`, `label`, `notify_url`, `created` and
`jobs`, the job objects.
`GET {{BASE}}/api/batches/{batch_id}?wait=SECONDS` answers when every job of the batch has ended or after SECONDS
(0 to {{WAIT_MAX}}).

## Job history

`GET {{BASE}}/api/jobs?status=&job=&batch_id=&before=&limit=` lists job objects, newest first, as
`{"jobs": [...], "next_before": ...}`. `limit` is 1 to {{PAGE_MAX}}, default {{PAGE_SIZE}}. `next_before` is the `seq` to pass as
`before` for the next page; it is null on the last page.

## Cancel, retry, delete

- `POST {{BASE}}/api/jobs/{job_id}/cancel` cancels a queued or running job; answer 202 with the job object, which ends
  as `cancelled` after its running stage has stopped. A job that has ended is refused with 409.
- `POST {{BASE}}/api/jobs/{job_id}/retry` creates a new job with the same job, params, inputs, count, priority, label
  and notify_url, and answers 201 with it. The body `{"seed": "same"}` (default) keeps the seed; `{"seed": "new"}`
  draws a new random seed for a job that has a `seed` param. The inputs and the installation are checked again.
- `DELETE {{BASE}}/api/jobs/{job_id}` deletes a job that has ended, with its output files. It is refused with 409 while
  an output of the job is an input of a queued or running job.

## Notifications

When a job ends, and when every job of a batch has ended, the factory sends one POST with the JSON
`{"from": "asset-factory", "text": "..."}` to the `notify_url`. The text holds the status, the error, the `file_id` and
the path of each output file, and "work took N s (X $)". A connection error, a timeout or an HTTP status of 500 or more
is retried at growing intervals while the factory runs; any other status ends the delivery. A job that was queued or
running when the factory stopped is reported as failed with "factory stopped" when the factory starts.

## Errors

Every refusal and every validation error is `{"error": {"message": "...", "details": ["...", ...]}}`. `message` names
the state and the next action; `details` lists every defect, each starting with the path of the field it is about.
- 400: the request is invalid.
- 404: an id does not exist, or a GET goes to a path that is not an endpoint.
- 405: a request other than GET goes to a path with no endpoint for its method.
- 409: the state of the factory does not allow the request: a job that is not installed, a job that has ended or has not
  ended, a busy queue.

## System

- `GET {{BASE}}/api/health` answers `{"status": "ok"}`.
- `GET {{BASE}}/api/system` answers the GPU (`name`, `vram_used_gb`, `vram_total_gb`), the RAM (`total_gb`,
  `available_gb`), `local_queue` and `cloud_queue`. The local queue runs one stage at a time on the machine and keeps
  at most one Worker (a process with loaded models) alive; an idle Worker is unloaded after a timeout. Its `worker` is
  the alive Worker, `running` the stage being run, `waiting` the stages in the order they are served and `ram_waits`
  those that wait for system RAM. The cloud queue has one lane per Modal app, with the call being computed,
  the waiting calls, and the cloud `seconds` and `dollars` since the factory started.
- `POST {{BASE}}/api/system/unload` stops the alive Worker and answers the local queue. It is refused with 409 while a
  stage is starting or running.
- `GET {{BASE}}/api/events` is a server-sent event stream. Each `data:` line is a job object, sent when the job is
  submitted, starts, reports progress or ends. When a job is deleted, the stream sends the event `deleted` whose
  `data:` line is the JSON string of its `job_id`.

## Job catalog

Each section below is one job: its description, the JSON schema of its params, its inputs with their kind and number of
files, and its output files by name.

