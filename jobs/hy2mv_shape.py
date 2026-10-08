from typing import Literal

from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class Hy2mvShape(Job):
    class Params(BaseParams):
        front: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        left: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"], 0, 1)
        back: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"], 0, 1)
        right: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"], 0, 1)
        steps: int = Field(ge=1, le=30)
        guidance_scale: float = Field(ge=1.0, le=15.0)
        octree_resolution: Literal[128, 192, 256, 384]
        num_chunks: int = Field(ge=1000, le=200000)
        target_faces: int
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        sent = {name: paths[0] for name, paths in ctx.files.items() if paths}
        images = (await ctx.local("bgremove", images=list(sent.values())))["images"]
        views = dict(zip(sent, images))
        raw = await ctx.local("hy2mv", front=views["front"], left=views.get("left"), back=views.get("back"),
                              right=views.get("right"), steps=p.steps, guidance_scale=p.guidance_scale,
                              octree_resolution=p.octree_resolution, num_chunks=p.num_chunks, seed=ctx.seed)
        mesh = (await ctx.local("clean", mesh=raw["mesh"]))["mesh"]
        if p.target_faces > 0:
            mesh = (await ctx.local("decimate", mesh=mesh, target_faces=p.target_faces))["mesh"]
        ctx.output(mesh)
        for image in images:
            ctx.output(image)
