from core.job import BaseParams, Input, Job, Seed
from pydantic import Field


class Hy2mvShape(Job):
    class Params(BaseParams):
        steps: int = Field(ge=1, le=30)
        guidance_scale: float = Field(ge=1.0, le=15.0)
        octree_resolution: int = Field(ge=64, le=384)
        num_chunks: int = Field(ge=1000, le=200000)
        target_faces: int
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
        mesh = (await ctx.local("clean", mesh=raw["mesh"]))["mesh"]
        if p.target_faces > 0:
            mesh = (await ctx.local("decimate", mesh=mesh, target_faces=p.target_faces))["mesh"]
        ctx.output(mesh, "mesh")
        for image in images:
            ctx.output(image, "image")
