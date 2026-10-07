from typing import Literal

from core.job import BaseParams, Input, Job, Seed
from pydantic import Field


class Trellis2Shape(Job):
    class Params(BaseParams):
        resolution: Literal[512, 1024, 1536]
        steps: int = Field(ge=1, le=50)
        target_faces: int = Field(gt=0)
        seed: Seed

    inputs = {"image": Input("image")}

    async def run(self, ctx):
        p = ctx.params
        raw = await ctx.cloud("trellis2", {"image": ctx.file("image")},
                              {"resolution": p.resolution, "steps": p.steps, "target_faces": p.target_faces,
                               "seed": ctx.seed})
        cleaned = (await ctx.local("clean", mesh=raw["raw.glb"]))["mesh"]
        mesh = (await ctx.local("decimate", mesh=cleaned, target_faces=p.target_faces))["mesh"]
        ctx.output(mesh, "mesh")
