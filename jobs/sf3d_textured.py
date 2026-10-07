from typing import Literal

from core.job import BaseParams, Input, Job, Seed
from pydantic import Field


class Sf3dTextured(Job):
    class Params(BaseParams):
        texture_resolution: Literal[256, 512, 1024, 2048]
        remesh: Literal["none", "triangle", "quad"]
        target_vertex_count: int = Field(ge=-1, le=100000)
        foreground_ratio: float = Field(ge=0.5, le=1.0)
        input_elevation_deg: float = Field(ge=-30, le=60)
        seed: Seed

    inputs = {"image": Input("image")}

    async def run(self, ctx):
        p = ctx.params
        image = (await ctx.local("bgremove", images=[ctx.file("image")]))["images"][0]
        result = await ctx.local("sf3d", image=image, texture_resolution=p.texture_resolution, remesh=p.remesh,
                                 target_vertex_count=p.target_vertex_count, foreground_ratio=p.foreground_ratio,
                                 input_elevation_deg=p.input_elevation_deg, seed=ctx.seed)
        ctx.output(result["mesh"], "mesh")
        ctx.output(image, "image")
