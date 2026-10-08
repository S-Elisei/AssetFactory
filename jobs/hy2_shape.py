from typing import Literal

from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class Hy2Shape(Job):
    class Params(BaseParams):
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        steps: int = Field(ge=1, le=30)
        guidance_scale: float = Field(ge=1.0, le=15.0)
        octree_resolution: Literal[128, 192, 256, 384]
        num_chunks: int = Field(ge=1000, le=200000)
        target_faces: int
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        image = (await ctx.local("bgremove", images=[ctx.file("image")]))["images"][0]
        raw = await ctx.local("hy2", image=image, steps=p.steps, guidance_scale=p.guidance_scale,
                              octree_resolution=p.octree_resolution, num_chunks=p.num_chunks, seed=ctx.seed)
        mesh = (await ctx.local("clean", mesh=raw["mesh"]))["mesh"]
        if p.target_faces > 0:
            mesh = (await ctx.local("decimate", mesh=mesh, target_faces=p.target_faces))["mesh"]
        ctx.output(mesh)
        ctx.output(image)
