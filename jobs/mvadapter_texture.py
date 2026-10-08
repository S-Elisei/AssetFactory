import asyncio
from typing import Literal

import numpy as np
import trimesh
from core.job import BaseParams, Files, Job, Seed
from PIL import Image
from pydantic import Field
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


def _write_glb(path, source, base_color):
    """Writes the GLB of the mesh in the GLB `source` with its own UVs, the vertex normals welded by position, and a
    material of the base-color image `base_color`, with metallic factor 0 and roughness factor 1."""
    mesh = trimesh.load(source, file_type="glb", force="mesh", process=False)
    unique, inverse = np.unique(mesh.vertices + 0.0, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    normals = trimesh.Trimesh(unique, inverse[mesh.faces], process=False).vertex_normals[inverse]
    material = PBRMaterial(baseColorTexture=Image.open(base_color), metallicFactor=0.0, roughnessFactor=1.0)
    trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_normals=normals,
                    visual=TextureVisuals(uv=mesh.visual.uv, material=material), process=False
                    ).export(path, include_normals=True)


class MvadapterTexture(Job):
    class Params(BaseParams):
        mesh: Files([".glb"])
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        steps: int = Field(ge=4, le=50)
        guidance_scale: float = Field(ge=1.0, le=10.0)
        texture_resolution: Literal[1024, 2048]
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        mesh = ctx.file("mesh")
        image = (await ctx.local("bgremove", images=[ctx.file("image")]))["images"][0]
        views = await ctx.local("mvadapter", mesh=mesh, image=image, steps=p.steps, guidance_scale=p.guidance_scale,
                                texture_resolution=p.texture_resolution, seed=ctx.seed)
        baked = await ctx.local("bake", mesh=mesh, view_sets=[views["views"]], cameras=views["cameras"],
                                texture_size=p.texture_resolution)
        filled = await ctx.local("fill", mesh=mesh, covered=baked["covered"], valid=baked["valid"],
                                 sets=[{"atlas": baked["atlases"][0], "views": views["views"]}], seed=ctx.seed)
        atlas = filled["atlases"][0]
        glb = ctx.dir / "mesh.glb"
        await asyncio.to_thread(_write_glb, glb, mesh, atlas)
        ctx.output(glb)
        ctx.output(atlas)
        ctx.output(image)
