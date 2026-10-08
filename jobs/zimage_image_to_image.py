from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class ZimageImageToImage(Job):
    class Params(BaseParams):
        image: Files([".png", ".jpg", ".jpeg", ".webp", ".bmp"])
        prompt: str = Field(max_length=2000)
        strength: float = Field(ge=0.1, le=1.0)
        steps: int = Field(ge=4, le=16)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("zimage", prompt=p.prompt, steps=p.steps, seed=ctx.seed, width=None, height=None,
                                 image=ctx.file("image"), strength=p.strength)
        ctx.output(result["image"])
