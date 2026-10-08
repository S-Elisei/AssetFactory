from typing import Literal

from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class TriposgShape(Job):
    class Params(BaseParams):
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        steps: int = Field(ge=8, le=100)
        guidance_scale: float = Field(ge=0.0, le=20.0)
        octree_depth: Literal[8, 9]
        target_faces: int
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        image = (await ctx.local("bgremove", images=[ctx.file("image")]))["images"][0]
        raw = await ctx.local("triposg", image=image, steps=p.steps, guidance_scale=p.guidance_scale,
                              octree_depth=p.octree_depth, seed=ctx.seed)
        mesh = (await ctx.local("clean", mesh=raw["mesh"]))["mesh"]
        if p.target_faces > 0:
            mesh = (await ctx.local("decimate", mesh=mesh, target_faces=p.target_faces))["mesh"]
        ctx.output(mesh)
        ctx.output(image)
