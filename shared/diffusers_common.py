"""Code shared by the stages of the `diffusers` environment: fitting a text encoder, a transformer and a VAE onto the
GPU, the prompt embeddings kept between runs, and the sampling-step callback. Imports torch, accelerate and
transformers."""
from pathlib import Path

import mapped
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3Model

DEVICE = torch.device("cuda")

# Size in bytes of one pinned host slab; a tensor larger than a slab gets a slab of its own size. Guessed.
SLAB_BYTES = 2**30

# (maximum output pixels, bytes of transformer groups kept on the GPU); an output with more pixels keeps none. Guessed.
RESIDENT_BUDGETS = ((300_000, 4 * 2**30), (650_000, 3 * 2**30))


def _weights_to_device(module, _args):
    mapped.to_device(module, DEVICE, recurse=False)


def _weights_to_host(module, _args, _output):
    mapped.to_host(module, recurse=False)


def load_text_encoder(snapshot, layers=None):
    """Returns the Qwen3 text encoder of the diffusers checkpoint folder `snapshot` with its first `layers` decoder
    layers (None: all), built without weights and given parameters that point to memory maps of the checkpoint files;
    the parameters of each submodule are copied to the GPU right before the submodule runs and point back to the maps
    after it returned. Its buffers are on the GPU."""
    directory = Path(snapshot) / "text_encoder"
    config = AutoConfig.from_pretrained(directory)
    if layers is not None:
        config.num_hidden_layers = layers
    with init_empty_weights():
        text_encoder = Qwen3Model(config)
    tensors = mapped.map_tensors(directory.glob("*.safetensors"))
    mapped.attach(text_encoder, {key: tensors["model." + key] for key in text_encoder.state_dict()})
    for buffer in text_encoder.buffers():
        buffer.data = buffer.data.to(DEVICE)
    for module in text_encoder.modules():
        if list(module.parameters(recurse=False)):
            module.register_forward_pre_hook(_weights_to_device)
            module.register_forward_hook(_weights_to_host)
    return text_encoder


def stream_transformer(transformer):
    """Moves the parameters and buffers of `transformer` into pinned host slabs, then enables diffusers group
    offloading: each block is copied to the GPU one block ahead of its use on a side CUDA stream. Returns the groups for
    keep_resident()."""
    slab, offset = None, 0
    for tensor in [*transformer.parameters(), *transformer.buffers()]:
        size = tensor.data.nbytes
        if slab is None or offset + size > slab.numel():
            slab = torch.empty(max(SLAB_BYTES, size), dtype=torch.uint8, pin_memory=True)
            offset = 0
        view = slab[offset:offset + size].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor.data)
        tensor.data = view
        offset = (offset + size + 255) // 256 * 256
    transformer.enable_group_offload(DEVICE, offload_type="block_level", num_blocks_per_group=1, use_stream=True)
    groups = [module._diffusers_hook.get_hook("group_offloading").group
              for module in transformer.modules() if hasattr(module, "_diffusers_hook")]
    return [(group, sum(parameter.nbytes for module in group.modules for parameter in module.parameters())
             + sum(parameter.nbytes for parameter in group.parameters)) for group in groups]


def resident_budget(pixels):
    """Returns the bytes of transformer groups that keep_resident() may hold on the GPU for an output of `pixels`."""
    return next((budget for limit, budget in RESIDENT_BUDGETS if pixels <= limit), 0)


def _nothing():
    pass


def keep_resident(groups, budget):
    """Keeps the leading groups of `groups` (from stream_transformer) that fit into `budget` bytes on the GPU; the other
    groups keep streaming. A resident group has no-op onload_ and offload_ set on the group object. Must not run while
    the transformer executes."""
    count = 0
    for _, size in groups:
        if size > budget:
            break
        budget -= size
        count += 1
    evicted = False
    for group, _ in groups[count:]:
        if "onload_" in vars(group):
            del group.onload_, group.offload_
            group.offload_()
            evicted = True
    if evicted:
        torch.cuda.empty_cache()
    loaded = False
    for group, _ in groups[:count]:
        if "onload_" not in vars(group):
            group.onload_()
            group.onload_ = group.offload_ = _nothing
            loaded = True
    if loaded:
        torch.cuda.synchronize()


def load_vae(vae_class, snapshot):
    """Returns the bf16 VAE of the checkpoint folder `snapshot` on the GPU; it encodes and decodes in tiles."""
    vae = vae_class.from_pretrained(snapshot, subfolder="vae", dtype=torch.bfloat16).to(DEVICE)
    vae.enable_tiling()
    return vae


def prompt_embeds(ctx, cache, pipeline, prompt, **options):
    """Returns the first result of `pipeline.encode_prompt(prompt, **options)`. `cache` is a dict that keeps the result
    of the last distinct `prompt`; the CUDA cache is emptied after an encoding."""
    if cache.get("prompt") != prompt:
        ctx.progress(0.0, "encoding prompt")
        with torch.no_grad():
            cache["embeds"] = pipeline.encode_prompt(prompt, **options)[0]
        cache["prompt"] = prompt
        torch.cuda.empty_cache()
    return cache["embeds"]


def sampling_callback(ctx):
    """Returns a `callback_on_step_end` for a diffusers pipeline: it raises Cancelled when the run was cancelled,
    reports the finished sampling step and, after the last step, empties the CUDA cache and reports the start of
    decoding."""

    def callback(pipeline, step, _timestep, _tensors):
        ctx.check_cancel()
        steps = pipeline.num_timesteps
        ctx.progress(0.9 * (step + 1) / steps, f"sampling step {step + 1}/{steps}")
        if step + 1 == steps:
            torch.cuda.empty_cache()
            ctx.progress(0.9, "decoding")
        return {}

    return callback
