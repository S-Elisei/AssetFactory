import asyncio
from typing import Annotated, Literal

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


class UnitexTexture(Job):
    class Params(BaseParams):
        mesh: Files([".glb"])
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        delight: bool
        texture_size: Literal[1024, 2048]
        atlas: Annotated[Literal["lit", "delit"], Field(json_schema_extra={"x-requires": {"delit": {"delight": True}}})]
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        mesh = ctx.file("mesh")
        generated = await ctx.cloud("unitex", {"mesh": mesh, "image": ctx.file("image")},
                                    {"delight": p.delight, "texture_size": p.texture_size, "seed": ctx.seed})
        view_sets = [[generated[f"{kind}_view_{number}.png"] for number in range(6)]
                     for kind in (["lit", "delit"] if p.delight else ["lit"])]
        baked = await ctx.local("bake", mesh=mesh, view_sets=view_sets, cameras=generated["cameras.json"],
                                texture_size=p.texture_size)
        filled = await ctx.local("fill", mesh=mesh, covered=baked["covered"], valid=baked["valid"],
                                 sets=[{"atlas": atlas, "views": views}
                                       for atlas, views in zip(baked["atlases"], view_sets)], seed=ctx.seed)
        glb = ctx.dir / "mesh.glb"
        await asyncio.to_thread(_write_glb, glb, mesh, filled["atlases"][1 if p.atlas == "delit" else 0])
        ctx.output(glb)
        for atlas in filled["atlases"]:
            ctx.output(atlas)
