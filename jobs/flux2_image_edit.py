from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class Flux2ImageEdit(Job):
    class Params(BaseParams):
        reference_images: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"], 1, 4)
        prompt: str = Field(max_length=2000)
        width: int = Field(ge=256, le=2048, multiple_of=16)
        height: int = Field(ge=256, le=2048, multiple_of=16)
        steps: int = Field(ge=2, le=8)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("flux2", prompt=p.prompt, width=p.width, height=p.height, steps=p.steps,
                                 seed=ctx.seed, reference_images=ctx.files["reference_images"])
        ctx.output(result["image"])
