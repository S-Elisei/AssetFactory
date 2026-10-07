from core.job import BaseParams, Job, Seed
from pydantic import Field


class ZimageTextToImage(Job):
    class Params(BaseParams):
        prompt: str = Field(max_length=2000)
        width: int = Field(ge=256, le=2048, multiple_of=16)
        height: int = Field(ge=256, le=2048, multiple_of=16)
        steps: int = Field(ge=4, le=16)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("zimage", prompt=p.prompt, steps=p.steps, seed=ctx.seed, width=p.width,
                                 height=p.height, image=None, strength=None)
        ctx.output(result["image"], "image")
