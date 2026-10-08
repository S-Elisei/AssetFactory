import asyncio
from typing import Literal

import numpy as np
import trimesh
from core.job import BaseParams, Files, Job, Seed
from PIL import Image
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


class HypaintTexture(Job):
    class Params(BaseParams):
        mesh: Files([".glb"])
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        texture_resolution: Literal[512, 1024, 2048]
        delight: bool
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        mesh = ctx.file("mesh")
        image = (await ctx.local("bgremove", images=[ctx.file("image")]))["images"][0]
        textures = await ctx.local("hypaint", mesh=mesh, image=image, normal_source=None,
                                   texture_resolution=p.texture_resolution, delight=p.delight, seed=ctx.seed)
        glb = ctx.dir / "mesh.glb"
        await asyncio.to_thread(_write_glb, glb, mesh, textures["base_color"])
        ctx.output(glb)
        ctx.output(textures["base_color"])
        ctx.output(image)
