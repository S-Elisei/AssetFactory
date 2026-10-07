import asyncio
from typing import Literal

import numpy as np
import trimesh
from core.job import BaseParams, Input, Job, Seed
from PIL import Image
from pydantic import Field
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


def _write_glb(path, source, base_color, normal_map):
    """Writes the GLB of the mesh in the GLB `source` with its own UVs, the vertex normals welded by position, and a
    material of the base-color image `base_color` and the normal image `normal_map` (None for no normal texture), with
    metallic factor 0 and roughness factor 1."""
    mesh = trimesh.load(source, file_type="glb", force="mesh", process=False)
    unique, inverse = np.unique(mesh.vertices + 0.0, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    normals = trimesh.Trimesh(unique, inverse[mesh.faces], process=False).vertex_normals[inverse]
    material = PBRMaterial(baseColorTexture=Image.open(base_color),
                           normalTexture=None if normal_map is None else Image.open(normal_map),
                           metallicFactor=0.0, roughnessFactor=1.0)
    trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_normals=normals,
                    visual=TextureVisuals(uv=mesh.visual.uv, material=material), process=False
                    ).export(path, include_normals=True)


class Hy2mvTextured(Job):
    class Params(BaseParams):
        steps: int = Field(ge=1, le=30)
        guidance_scale: float = Field(ge=1.0, le=15.0)
        octree_resolution: int = Field(ge=64, le=384)
        num_chunks: int = Field(ge=1000, le=200000)
        target_faces: int
        texture_resolution: Literal[512, 1024, 2048]
        delight: bool
        normal_map: bool
        seed: Seed

    inputs = {"front": Input("image"), "left": Input("image", 0, 1), "back": Input("image", 0, 1),
              "right": Input("image", 0, 1)}

    async def run(self, ctx):
        p = ctx.params
        sent = {name: paths[0] for name, paths in ctx.inputs.items() if paths}
        images = (await ctx.local("bgremove", images=list(sent.values())))["images"]
        views = dict(zip(sent, images))
        raw = await ctx.local("hy2mv", front=views["front"], left=views.get("left"), back=views.get("back"),
                              right=views.get("right"), steps=p.steps, guidance_scale=p.guidance_scale,
                              octree_resolution=p.octree_resolution, num_chunks=p.num_chunks, seed=ctx.seed)
        cleaned = (await ctx.local("clean", mesh=raw["mesh"]))["mesh"]
        mesh = cleaned
        if p.target_faces > 0:
            mesh = (await ctx.local("decimate", mesh=cleaned, target_faces=p.target_faces))["mesh"]
        unwrapped = (await ctx.local("unwrap", mesh=mesh))["mesh"]
        textures = await ctx.local("hypaint", mesh=unwrapped, image=views["front"],
                                   normal_source=cleaned if p.normal_map else None,
                                   texture_resolution=p.texture_resolution, delight=p.delight, seed=ctx.seed)
        glb = ctx.dir / "mesh.glb"
        await asyncio.to_thread(_write_glb, glb, unwrapped, textures["base_color"], textures["normal_map"])
        ctx.output(glb, "mesh")
        ctx.output(textures["base_color"], "image")
        if p.normal_map:
            ctx.output(textures["normal_map"], "image")
        for image in images:
            ctx.output(image, "image")
