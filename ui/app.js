"use strict";

// ---------- helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (k === "selected") el.selected = true;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

class ApiFailure extends Error {
  constructor(status, error) { super(error.message); this.status = status; this.error = error; }
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new ApiFailure(r.status, data.error);
  return data;
}

function toast(msg, kind = "") {
  const t = h("div", { class: "toast " + kind, text: msg });
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 9000 : 4500);
}

function fail(e) { toast(e instanceof ApiFailure ? e.error.message : String(e), "error"); }

const badge = (text) => h("span", { class: `badge b-${text}`, text });
const fmtTime = (sec) => (sec ? new Date(sec * 1000).toLocaleString() : "");
const fmtBytes = (n) => (n < 1024 ? `${n} B` : n < 2 ** 20 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 2 ** 20).toFixed(1)} MB`);
const contentUrl = (f) => `/api/files/${f.file_id}/content`;
const extOf = (name) => name.slice(name.lastIndexOf(".")).toLowerCase();
const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];
const AUDIO_EXTS = [".wav", ".flac", ".ogg", ".mp3"];
const isImage = (f) => IMAGE_EXTS.includes(extOf(f.name));
const extPlaceholder = (f, style) => h("div", { class: "ph", style, text: extOf(f.name) });

function openModal(body, cls = "") {
  S.back = null;
  $("#modal .modal-box").className = "modal-box" + (cls ? " " + cls : "");
  $("#modal-body").replaceChildren(body);
  $("#modal").classList.remove("hidden");
}
// Closing a modal that was opened from the New form returns to the New form.
function closeModal() {
  if (S.back) { S.back(); return; }
  $("#modal").classList.add("hidden");
  $("#modal-body").replaceChildren();
}

// ---------- state ----------
const S = {
  models: [],                 // models.json: [{name, tasks: [{task, job, values}]}]
  schema: {},                 // GET /api/jobs/schema: job name -> {description, modality, params, outputs}
  byJob: {},                  // job name -> {model, task}
  queued: new Map(),          // job_id -> queued job object
  live: new Map(),            // job_id -> status of the jobs that have not ended
  system: null,
  files: new Map(),           // file_id -> Promise of the file object, null for a deleted file
  gen: { modality: "all", model: null, task: null, values: {}, files: {}, count: 1, seed: null },
  hist: { model: "", task: "", status: "", jobs: [], next: null, token: 0 },
  back: null,                 // called by closeModal instead of closing
};

const viewOf = (job) => S.byJob[job.job];
const hasSeed = (jobName) => "seed" in S.schema[jobName].params.properties;

function fileOf(id) {
  if (!S.files.has(id)) {
    S.files.set(id, api("GET", `/api/files/${id}`).catch((e) => {
      if (e instanceof ApiFailure && e.status === 404) return null;
      throw e;
    }));
  }
  return S.files.get(id);
}

// ---------- header ----------
function setLoad(id, fraction, text) {
  const el = $("#" + id);
  $(".bar > div", el).style.width = `${Math.min(100, Math.max(0, fraction * 100))}%`;
  $("span", el).textContent = text;
}

function updateHeader(s) {
  const ramUsed = s.ram.total_gb - s.ram.available_gb;
  setLoad("load-ram", ramUsed / s.ram.total_gb, `${ramUsed.toFixed(1)} / ${s.ram.total_gb.toFixed(1)} GB`);
  setLoad("load-vram", s.gpu.vram_used_gb / s.gpu.vram_total_gb, `${s.gpu.vram_used_gb.toFixed(1)} / ${s.gpu.vram_total_gb.toFixed(1)} GB`);
  setLoad("load-cpu", s.cpu.percent / 100, `${Math.round(s.cpu.percent)}%`);
  const gpu = s.gpu.utilization_percent;
  setLoad("load-gpu", gpu === null ? 0 : gpu / 100, gpu === null ? "—" : `${gpu}%`);
  $("#load-gpu").title = s.gpu.name;
  const dollars = (x) => (x === null ? "…" : `$${x.toFixed(2)}`);
  $("#credits").textContent = `Modal credits: ${dollars(s.cloud.credits_used_dollars)} used, ${dollars(s.cloud.billed_dollars)} billed`;
}

async function pollSystem() {
  S.system = await api("GET", "/api/system");
  updateHeader(S.system);
  for (const job of S.hist.jobs) if (job.status === "running") patchJobCard(job);
}

// ---------- live updates ----------
function connect() {
  const es = new EventSource("/api/events");
  es.onopen = () => refreshAll().catch(fail);
  es.onmessage = (e) => onJob(JSON.parse(e.data));
  es.addEventListener("deleted", (e) => onDeleted(JSON.parse(e.data)));
}

async function refreshAll() {
  const [queued, running] = await Promise.all([
    api("GET", "/api/jobs?status=queued&limit=200"), api("GET", "/api/jobs?status=running&limit=200")]);
  S.queued = new Map(queued.jobs.map((j) => [j.job_id, j]));
  S.live = new Map([...queued.jobs, ...running.jobs].map((j) => [j.job_id, j.status]));
  renderQueue();
  await loadHistory(false);
}

function onJob(job) {
  const was = S.live.get(job.job_id);
  if (job.status === "queued" || job.status === "running") {
    S.live.set(job.job_id, job.status);
  } else {
    S.live.delete(job.job_id);
    if (was) toast(`${job.job_id} ${job.status}${job.error ? ": " + job.error.message : ""}`, job.status === "succeeded" ? "ok" : job.status === "failed" ? "error" : "");
  }
  if (job.status === "queued") { S.queued.set(job.job_id, job); renderQueue(); return; }
  if (S.queued.delete(job.job_id)) renderQueue();
  updateHistory(job);
}

function onDeleted(jobId) {
  S.live.delete(jobId);
  if (S.queued.delete(jobId)) renderQueue();
  S.hist.jobs = S.hist.jobs.filter((j) => j.job_id !== jobId);
  document.getElementById("job-" + jobId)?.remove();
  for (const [id, file] of S.files) file.then((f) => { if (f && f.job_id === jobId) S.files.delete(id); });
}

// ---------- new ----------
function randomSeed(props) {
  const max = props.seed.anyOf.find((x) => x.type === "integer").maximum;
  return Math.floor(Math.random() * (max + 1));
}

const filesOf = (job, name) => (S.gen.files[`${job}/${name}`] ||= []);

function fieldWidget(name, spec, values) {
  const set = (x) => { values[name] = x; };
  let input;
  if (spec.enum || spec.type === "boolean") {
    const options = spec.enum || [true, false];
    const allowed = (o) => Object.entries((spec["x-requires"] || {})[o] || {}).every(([other, v]) => values[other] === v);
    if (values[name] === undefined || !allowed(values[name])) values[name] = options.find(allowed);
    input = h("select", { onchange: (e) => {
      const raw = e.target.value;
      set(spec.type === "string" ? raw : spec.type === "boolean" ? raw === "true" : Number(raw));
      renderNew();
    } }, options.map((o) => h("option", { value: String(o), selected: String(o) === String(values[name]), disabled: !allowed(o), text: String(o) })));
  } else if (spec.type === "string") {
    if (values[name] === undefined) values[name] = "";
    input = h("textarea", { value: values[name], maxlength: spec.maxLength, oninput: (e) => set(e.target.value) });
  } else {
    const step = spec.multipleOf || (spec.type === "integer" ? 1 : "any");
    input = h("input", { type: "number", value: values[name] ?? "", min: spec.minimum, max: spec.maximum, step, required: true,
      oninput: (e) => set(Number(e.target.value)) });
  }
  const range = spec.minimum !== undefined || spec.maximum !== undefined ? ` (${spec.minimum ?? ""}..${spec.maximum ?? ""})` : "";
  return h("label", { class: "field" + (spec.type === "string" && !spec.enum ? "" : " inline"), "data-field": `params.${name}` },
    h("span", { class: "name" }, name, h("span", { class: "muted small", text: ` ${spec.type}${range}${spec.multipleOf ? ", multiple of " + spec.multipleOf : ""}` })),
    input, h("span", { class: "desc", text: spec.description }), h("span", { class: "ferr" }));
}

function fileThumb(f, onRemove) {
  const vis = isImage(f) ? h("img", { src: contentUrl(f), alt: f.name }) : extPlaceholder(f, "height:72px;display:flex;align-items:center;justify-content:center");
  return h("div", { class: "thumb", title: f.name }, vis, h("div", { class: "n", text: f.name }),
    onRemove ? h("button", { class: "ghost small x", type: "button", text: "x", onclick: onRemove }) : null);
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file, file.name);
  return api("POST", "/api/files", fd);
}

// Opened from the New form: closing it, or picking a file, returns to the form with its scroll position.
function pickFile(exts, onPick) {
  const scroll = $("#modal .modal-box").scrollTop;
  const grid = h("div", { class: "pickgrid" }, h("div", { class: "muted", text: "Loading..." }));
  openModal(grid);
  S.back = () => renderNew(scroll);
  api("GET", "/api/files").then((r) => {
    const files = r.files.filter((f) => exts.includes(extOf(f.name)));
    grid.replaceChildren(...(files.length ? files.map((f) => h("div", { class: "pick", title: f.name, onclick: () => { onPick(f); renderNew(scroll); } },
      isImage(f) ? h("img", { src: contentUrl(f), alt: f.name, loading: "lazy" }) : extPlaceholder(f),
      h("div", { text: f.name }), h("div", { class: "muted", text: f.job_id ?? f.origin }))) : [h("div", { class: "muted", text: `No ${exts.join(" ")} files yet.` })]));
  }, fail);
}

function fileField(name, spec, files) {
  const exts = spec["x-extensions"];
  const max = spec.maxItems;
  const box = h("div", { class: "field", "data-field": `params.${name}` });
  const rerender = () => box.replaceWith(fileField(name, spec, files));
  const push = (f) => {
    if (files.length >= max) files.splice(0, 1);
    files.push(f);
  };
  const addLocal = async (list) => {
    let added = false;
    for (const file of list) {
      if (!exts.includes(extOf(file.name))) { toast(`${file.name} is not accepted; ${name} takes ${exts.join(" ")}`, "error"); continue; }
      try { push(await uploadFile(file)); added = true; } catch (e) { fail(e); }
    }
    if (added) rerender();
  };
  const chooser = h("input", { type: "file", class: "hidden", accept: exts.join(","), multiple: max > 1, onchange: (e) => addLocal([...e.target.files]) });
  const drop = h("div", { class: "drop",
    ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); },
    ondragleave: () => drop.classList.remove("over"),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove("over"); addLocal([...e.dataTransfer.files]); } },
    `Drop file${max > 1 ? "s" : ""} here or `,
    h("button", { class: "ghost small", type: "button", text: "Browse", onclick: () => chooser.click() }), " ",
    h("button", { class: "ghost small", type: "button", text: "Pick from outputs", onclick: () => pickFile(exts, push) }), chooser);
  box.append(
    h("span", { class: "name", style: "font-weight:600" }, name, h("span", { class: "muted small", text: ` ${exts.join(" ")}, ${spec.minItems}..${max} file(s)` })),
    h("div", { class: "desc muted small", text: spec.description }),
    h("div", { class: "row" }, h("div", { class: "thumbs" }, files.map((f, i) => fileThumb(f, () => { files.splice(i, 1); rerender(); }))), drop),
    h("span", { class: "ferr" }));
  return box;
}

function openNew() {
  S.gen.seed = null;
  renderNew();
}

const modalityOf = (m) => S.schema[m.tasks[0].job].modality;

function renderNew(scroll = $("#modal .modal-box").scrollTop) {
  const g = S.gen;
  const chips = h("div", { class: "chips" }, ["all", ...new Set(S.models.map(modalityOf))].map((x) =>
    h("button", { class: "chip" + (g.modality === x ? " on" : ""), text: x, onclick: () => { g.modality = x; renderNew(); } })));
  const shown = S.models.filter((m) => g.modality === "all" || modalityOf(m) === g.modality);
  const grid = h("div", { class: "modelgrid" }, shown.map((m) => h("div", {
    class: "mcard" + (g.model === m.name ? " sel" : ""), onclick: () => { g.model = m.name; g.task = m.tasks[0].task; renderNew(); } },
    h("div", { class: "t", text: m.name }), h("div", { class: "muted small", text: m.tasks.map((t) => t.task).join(", ") }))));
  openModal(h("div", { class: "cols" },
    h("div", {}, h("div", { class: "panel" }, h("h2", { text: "1. Choose a model" }), chips, grid)),
    h("div", {}, generateForm())), "wide");
  $("#modal .modal-box").scrollTop = scroll;
}

function generateForm() {
  const g = S.gen;
  const m = S.models.find((x) => x.name === g.model);
  const t = m.tasks.find((x) => x.task === g.task);
  const info = S.schema[t.job];
  const props = info.params.properties;
  const values = (g.values[t.job] ||= { ...t.values });
  const seeded = hasSeed(t.job);
  if (seeded && g.seed === null) g.seed = String(randomSeed(props));
  const errBox = h("div", { class: "err" });
  const fields = Object.entries(props).filter(([n]) => n !== "seed").map(([n, spec]) =>
    spec["x-extensions"] ? fileField(n, spec, filesOf(t.job, n)) : fieldWidget(n, spec, values));
  const seedInput = seeded ? h("input", { type: "number", min: 0, step: 1, value: g.seed, oninput: (e) => { g.seed = e.target.value; } }) : null;
  const form = h("form", { class: "panel", onsubmit: (e) => { e.preventDefault(); submit(t, props, values, form, errBox); } },
    h("h2", { text: "2. Describe the asset" }),
    h("div", { class: "row field" },
      m.tasks.length > 1 ? h("select", { onchange: (e) => { g.task = e.target.value; renderNew(); } },
        m.tasks.map((x) => h("option", { value: x.task, selected: x.task === g.task, text: x.task }))) : null,
      h("span", { class: "muted small grow", text: info.description })),
    fields,
    h("div", { class: "row", style: "margin-top:10px" },
      h("label", {}, "Count ", h("input", { type: "number", min: 1, value: g.count, style: "width:70px", oninput: (e) => { g.count = Number(e.target.value); } })),
      seeded ? h("label", { "data-field": "params.seed" }, "Seed ", seedInput, h("span", { class: "ferr" })) : null,
      seeded ? h("button", { class: "ghost small", type: "button", text: "Random", onclick: () => { g.seed = String(randomSeed(props)); seedInput.value = g.seed; } }) : null,
      h("button", { type: "submit", style: "margin-left:auto", text: "Generate" })),
    errBox);
  return form;
}

async function submit(t, props, values, form, errBox) {
  const g = S.gen;
  const params = {};
  for (const [name, spec] of Object.entries(props)) {
    if (spec["x-extensions"]) params[name] = filesOf(t.job, name).map((f) => f.file_id);
    else if (name === "seed") params.seed = g.seed === "" ? "random" : Number(g.seed);
    else params[name] = values[name];
  }
  for (const el of form.querySelectorAll(".invalid")) el.classList.remove("invalid");
  for (const el of form.querySelectorAll(".ferr")) el.textContent = "";
  errBox.replaceChildren();
  try {
    const job = await api("POST", "/api/jobs", { job: t.job, params, count: g.count, notify_url: null });
    closeModal();
    toast(`Queued ${job.job_id} (${job.count} item(s))`, "ok");
  } catch (e) {
    if (!(e instanceof ApiFailure)) throw e;
    const rest = [];
    for (const d of e.error.details) {
      const m = /^params\.(\w+)\S*: (.*)$/s.exec(d);
      const el = m && form.querySelector(`[data-field="params.${m[1]}"]`);
      if (!el) { rest.push(d); continue; }
      el.classList.add("invalid");
      const ferr = $(".ferr", el);
      ferr.textContent = ferr.textContent ? `${ferr.textContent}; ${m[2]}` : m[2];
    }
    errBox.replaceChildren(h("div", { text: e.error.message }), ...rest.map((d) => h("div", { text: d })));
  }
}

function useAsInput(f) {
  const g = S.gen;
  const fileParam = (job) => Object.entries(S.schema[job].params.properties).find(([, spec]) => spec["x-extensions"]?.includes(extOf(f.name)));
  const pick = (m) => m.tasks.find((t) => t.task === g.task && fileParam(t.job)) || m.tasks.find((t) => fileParam(t.job));
  const current = S.models.find((m) => m.name === g.model);
  const model = pick(current) ? current : S.models.find((m) => pick(m));
  const task = pick(model);
  g.model = model.name;
  g.task = task.task;
  const [name, spec] = fileParam(task.job);
  const files = filesOf(task.job, name);
  if (files.length >= spec.maxItems) files.splice(0, 1);
  files.push(f);
  openNew();
}

// ---------- job cards ----------
function promptOf(job) {
  const p = job.params;
  return p.prompt || p.text || p.caption || "";
}

// Materials without a base-color texture are shown as matte grey.
function greyUntextured(ev) {
  for (const m of ev.target.model.materials) {
    const pbr = m.pbrMetallicRoughness;
    if (pbr.baseColorTexture.texture) continue;
    pbr.setBaseColorFactor([0.18, 0.18, 0.18, 1]);
    pbr.setMetallicFactor(0);
    pbr.setRoughnessFactor(0.8);
  }
}

// Zoom: fitted into the window; a click shows the natural size, which a drag pans; another click fits again.
function openZoom(f) {
  const img = h("img", { class: "full", src: contentUrl(f), alt: f.name, draggable: "false" });
  openModal(img, "zoom");
  const body = $("#modal-body");
  let actual = false, x = 0, y = 0, drag = null, dragged = false;
  const place = () => { img.style.transform = `translate(${x}px, ${y}px)`; };
  const clamp = (v, box, natural) => Math.min(0, Math.max(box - natural, v));
  const downscaled = () => img.naturalWidth > img.clientWidth || img.naturalHeight > img.clientHeight;
  img.addEventListener("pointerenter", () => img.classList.toggle("zoomable", !actual && downscaled()));
  img.addEventListener("click", () => {
    if (dragged) return;
    if (actual) {
      actual = false;
      img.classList.remove("actual");
      img.classList.add("zoomable");
      img.style.transform = "";
    } else if (downscaled()) {
      actual = true;
      img.classList.remove("zoomable");
      img.classList.add("actual");
      x = Math.min(0, (body.clientWidth - img.naturalWidth) / 2);
      y = Math.min(0, (body.clientHeight - img.naturalHeight) / 2);
      place();
    }
  });
  img.addEventListener("pointerdown", (e) => {
    if (!actual) return;
    drag = { px: e.clientX, py: e.clientY, x, y };
    dragged = false;
    img.setPointerCapture(e.pointerId);
  });
  img.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.px, dy = e.clientY - drag.py;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragged = true;
    x = clamp(drag.x + dx, body.clientWidth, img.naturalWidth);
    y = clamp(drag.y + dy, body.clientHeight, img.naturalHeight);
    place();
  });
  img.addEventListener("pointerup", () => { drag = null; if (dragged) setTimeout(() => { dragged = false; }); });
  img.addEventListener("pointercancel", () => { drag = null; });
}

function outputView(f, job, item) {
  const url = contentUrl(f);
  let vis;
  if (isImage(f)) {
    vis = h("img", { src: url, alt: f.name, loading: "lazy", onclick: () => openZoom(f) });
  } else if (AUDIO_EXTS.includes(extOf(f.name))) {
    vis = h("audio", { controls: true, preload: "none", src: url });
  } else {
    vis = h("div", { class: "meshph" }, h("button", { class: "ghost small", text: "View 3D", onclick: () =>
      vis.replaceWith(h("model-viewer", { src: url, "camera-controls": true, "auto-rotate": true, "shadow-intensity": "0.6", alt: f.name, onload: greyUntextured })) }));
  }
  return h("div", { class: "out" }, vis,
    h("div", { class: "mono", title: f.name, style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap", text: f.name }),
    h("div", { class: "muted", text: `${fmtBytes(f.size)}${"seed" in job.params ? " · seed " + (job.params.seed + item) : ""}` }),
    h("div", { class: "acts" },
      h("a", { class: "btn ghost small", href: url, download: f.name, text: "Download" }),
      h("button", { class: "ghost small", text: "Use as input", onclick: () => useAsInput(f) })));
}

function outputsView(job) {
  const box = h("div", { class: "outputs" });
  const items = job.outputs.flatMap((ids, item) => ids.map((id) => [id, item]));
  Promise.all(items.map(([id]) => fileOf(id))).then((files) =>
    box.replaceChildren(...files.map((f, i) => outputView(f, job, items[i][1]))));
  return box;
}

async function showJobDetails(jobId) {
  let job;
  try { job = await api("GET", `/api/jobs/${jobId}`); } catch (e) { fail(e); return; }
  const view = viewOf(job);
  const p = job.params;
  const inputs = [];
  for (const [name, spec] of Object.entries(S.schema[job.job].params.properties)) {
    if (!spec["x-extensions"] || !p[name].length) continue;
    const files = await Promise.all(p[name].map(fileOf));
    inputs.push(h("div", {}, h("b", { text: name + ": " }),
      h("div", { class: "thumbs" }, files.map((f, i) => (f ? fileThumb(f) : h("span", { class: "muted", text: `${p[name][i]} (deleted)` }))))));
  }
  const rows = [
    ["job", h("div", { class: "mono", text: job.job_id })],
    job.batch_id ? ["batch", h("div", { class: "mono", text: job.batch_id })] : null,
    ["model", h("div", { text: view.model })],
    ["task", h("div", { text: view.task })],
    ["status", h("div", {}, badge(job.status))],
    "seed" in p ? [job.count > 1 ? "seeds" : "seed", h("div", { text: job.count > 1 ? `${p.seed}–${p.seed + job.count - 1}` : p.seed })] : null,
    ["count", h("div", { text: job.count })],
    ["created", h("div", { text: fmtTime(job.created) })],
    ["started", h("div", { text: fmtTime(job.started) })],
    ["finished", h("div", { text: fmtTime(job.finished) })],
    ["work seconds", h("div", { text: job.work_seconds.toFixed(1) })],
    ["cloud dollars", h("div", { text: job.cloud_dollars })],
  ].filter(Boolean);
  openModal(h("div", {},
    h("div", { class: "kv" }, rows.flatMap(([k, v]) => [h("div", { text: k }), v])),
    h("h3", { text: "Params" }), h("pre", { text: JSON.stringify(Object.fromEntries(Object.entries(p).filter(([k]) => k !== "seed")), null, 2) }),
    inputs.length ? h("h3", { text: "Inputs" }) : null, inputs,
    job.error ? h("h3", { text: "Error" }) : null, job.error ? h("pre", { text: JSON.stringify(job.error, null, 2) }) : null));
}

async function jobAction(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg(r), "ok"); } catch (e) { fail(e); }
}

function cardKey(job) {
  return `${job.status}/${job.outputs.flat().length}`;
}

// The step lines of a running job: the message of each stage call that runs for it, else the progress message.
function stepLines(job) {
  const s = S.system;
  const calls = [];
  if (s.local_queue.running && s.local_queue.running.tag === job.job_id) calls.push(["local", s.local_queue.running.message]);
  for (const lane of Object.values(s.cloud_queue)) {
    if (lane.running && lane.running.tag === job.job_id) calls.push(["cloud", lane.running.message]);
  }
  if (!calls.length) return [job.progress.message || "running"];
  return calls.map(([where, message]) => (calls.length === 1 ? message || "running" : `${where}: ${message || "running"}`));
}

const stepLine = (text) => h("div", { class: "muted small jmsg", text });

function metaText(job) {
  const p = job.params;
  const parts = [fmtTime(job.created), `work ${job.work_seconds.toFixed(1)} s`];
  if (job.cloud_dollars > 0) parts.push(`cloud $${job.cloud_dollars.toFixed(4)}`);
  parts.push(`${job.count} item(s)`);
  if ("seed" in p) parts.push(job.count > 1 ? `seeds ${p.seed}–${p.seed + job.count - 1}` : `seed ${p.seed}`);
  return parts.join(" · ");
}

function patchJobCard(job) {
  const el = document.getElementById("job-" + job.job_id);
  if (!el || el.dataset.key !== cardKey(job)) return false;
  if (job.status === "running") {
    $(".progress > div", el).style.width = `${Math.round(job.progress.fraction * 100)}%`;
    const lines = stepLines(job);
    const old = [...el.querySelectorAll(".jmsg")];
    if (old.length === lines.length) old.forEach((d, i) => { d.textContent = lines[i]; });
    else { old.forEach((d) => d.remove()); $(".progress", el).after(...lines.map(stepLine)); }
  }
  $(".head > .grow", el).textContent = metaText(job);
  return true;
}

function jobCard(job) {
  const active = job.status === "running";
  const view = viewOf(job);
  const act = (path, okMsg) => jobAction(() => api("POST", `/api/jobs/${job.job_id}/${path}`), okMsg);
  const details = h("button", { class: "ghost small", text: "Details", onclick: () => showJobDetails(job.job_id) });
  return h("div", { class: "job", id: "job-" + job.job_id, "data-key": cardKey(job) },
    h("div", { class: "head" }, badge(job.status), h("span", { class: "mono", text: job.job_id }),
      h("b", { text: view.model }), h("span", { class: "muted", text: view.task }),
      h("span", { class: "muted small grow", style: "text-align:right", text: metaText(job) })),
    promptOf(job) ? h("div", { class: "prompt", text: promptOf(job) }) : null,
    active ? h("div", { class: "progress" }, h("div", { style: `width:${Math.round(job.progress.fraction * 100)}%` })) : null,
    active ? stepLines(job).map(stepLine) : null,
    job.error ? h("div", { class: "err" }, h("b", { text: job.error.kind + ": " }), job.error.message) : null,
    job.outputs.some((ids) => ids.length) ? outputsView(job) : null,
    h("div", { class: "row", style: "margin-top:8px" },
      active ? h("button", { class: "danger small", text: "Cancel", onclick: () => act("cancel") }) : null,
      !active && (job.status !== "succeeded" || hasSeed(job.job)) ? h("button", { class: "ghost small", text: "Retry", onclick: () => act("retry", (r) => `Retry queued: ${r.job_id}`) }) : null,
      details,
      !active ? h("button", { class: "danger small", text: "Delete", onclick: () => {
        if (confirm(`Delete job ${job.job_id} and its output files?`)) jobAction(() => api("DELETE", `/api/jobs/${job.job_id}`));
      } }) : null));
}

// ---------- queue ----------
function renderQueue() {
  const jobs = [...S.queued.values()].sort((a, b) => a.seq - b.seq);
  $("#queue-title").textContent = `Queue (${jobs.length})`;
  $("#queue-list").replaceChildren(...(jobs.length ? jobs.map((job) => h("div", { class: "qrow" },
    h("b", { text: viewOf(job).model }), h("span", { class: "muted small grow", text: viewOf(job).task }),
    h("button", { class: "danger small", text: "×", onclick: () => jobAction(() => api("POST", `/api/jobs/${job.job_id}/cancel`)) })))
    : [h("div", { class: "muted", text: "Empty" })]));
}

// ---------- history ----------
function histJobs() {
  const hs = S.hist;
  if (!hs.model) return null;
  const tasks = S.models.find((m) => m.name === hs.model).tasks;
  return (hs.task ? tasks.filter((t) => t.task === hs.task) : tasks).map((t) => t.job);
}

function histMatches(job) {
  const names = histJobs();
  return (!names || names.includes(job.job)) && (!S.hist.status || job.status === S.hist.status);
}

async function loadHistory(more) {
  const hs = S.hist;
  const token = ++hs.token;
  const q = new URLSearchParams();
  const names = histJobs();
  if (names) q.set("job", names.join(","));
  if (hs.status) q.set("status", hs.status);
  if (more) q.set("before", hs.next);
  const r = await api("GET", "/api/jobs?" + q);
  if (token !== hs.token) return;
  const jobs = r.jobs.filter((j) => j.status !== "queued");
  hs.jobs = more ? hs.jobs.concat(jobs) : jobs;
  hs.next = r.next_before;
  if (more) $("#hist-list").append(...jobs.map(jobCard));
  else $("#hist-list").replaceChildren(...jobs.map(jobCard));
  $("#more").classList.toggle("hidden", hs.next === null);
}

function updateHistory(job) {
  const hs = S.hist;
  const i = hs.jobs.findIndex((j) => j.job_id === job.job_id);
  const el = document.getElementById("job-" + job.job_id);
  if (i >= 0) {
    if (!histMatches(job)) { hs.jobs.splice(i, 1); el.remove(); return; }
    hs.jobs[i] = job;
    if (!patchJobCard(job)) el.replaceWith(jobCard(job));
    return;
  }
  // A job older than the last row fetched (queued rows included) lies on a page that is not loaded.
  if (!histMatches(job) || (hs.next !== null && job.seq < hs.next)) return;
  const pos = hs.jobs.findIndex((j) => j.seq < job.seq);
  const at = pos < 0 ? hs.jobs.length : pos;
  hs.jobs.splice(at, 0, job);
  const list = $("#hist-list");
  list.insertBefore(jobCard(job), list.children[at] ?? null);
}

function fillTasks() {
  const model = S.models.find((m) => m.name === S.hist.model);
  $("#f-task").replaceChildren(h("option", { value: "", text: "all tasks" }),
    ...(model ? model.tasks.map((t) => h("option", { value: t.task, text: t.task })) : []));
  $("#f-task").disabled = !model;
}

// ---------- boot ----------
async function boot() {
  const [models, schema] = await Promise.all([fetch("models.json").then((r) => r.json()), api("GET", "/api/jobs/schema")]);
  S.models = models.models;
  S.schema = schema;
  for (const m of S.models) for (const t of m.tasks) S.byJob[t.job] = { model: m.name, task: t.task };
  S.gen.model = S.models[0].name;
  S.gen.task = S.models[0].tasks[0].task;
  $("#f-model").replaceChildren(h("option", { value: "", text: "all models" }), ...S.models.map((m) => h("option", { value: m.name, text: m.name })));
  fillTasks();
  await pollSystem();
  setInterval(pollSystem, 2000);
  connect();
}

$("#modal-close").addEventListener("click", closeModal);
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
$("#new").addEventListener("click", openNew);
$("#more").addEventListener("click", () => loadHistory(true).catch(fail));
$("#f-model").addEventListener("change", (e) => { S.hist.model = e.target.value; S.hist.task = ""; fillTasks(); loadHistory(false).catch(fail); });
$("#f-task").addEventListener("change", (e) => { S.hist.task = e.target.value; loadHistory(false).catch(fail); });
$("#f-status").addEventListener("change", (e) => { S.hist.status = e.target.value; loadHistory(false).catch(fail); });
$("#f-reset").addEventListener("click", () => {
  Object.assign(S.hist, { model: "", task: "", status: "" });
  $("#f-model").value = "";
  $("#f-status").value = "";
  fillTasks();
  loadHistory(false).catch(fail);
});
boot().catch(fail);
