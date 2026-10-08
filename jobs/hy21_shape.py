from typing import Literal

from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class Hy21Shape(Job):
    class Params(BaseParams):
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        steps: int = Field(ge=1, le=100)
        guidance_scale: float = Field(ge=1.0, le=15.0)
        octree_resolution: Literal[128, 192, 256, 384, 512]
        target_faces: int
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        raw = await ctx.cloud("hunyuan3d21", {"image": ctx.file("image")},
                              {"steps": p.steps, "guidance_scale": p.guidance_scale,
                               "octree_resolution": p.octree_resolution, "seed": ctx.seed})
        mesh = (await ctx.local("clean", mesh=raw["raw.glb"]))["mesh"]
        if p.target_faces > 0:
            mesh = (await ctx.local("decimate", mesh=mesh, target_faces=p.target_faces))["mesh"]
        ctx.output(mesh)
