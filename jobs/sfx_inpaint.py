from core.job import BaseParams, Input, Job, Seed
from pydantic import Field


class SfxInpaint(Job):
    class Params(BaseParams):
        prompt: str = Field(max_length=1000)
        start_seconds: float = Field(ge=0, le=120)
        end_seconds: float = Field(ge=0, le=120)
        steps: int = Field(ge=4, le=16)
        seed: Seed

    inputs = {"audio": Input("audio")}

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("stableaudio", prompt=p.prompt, steps=p.steps, seed=ctx.seed, duration=None,
                                 audio=ctx.file("audio"), noise_level=None, start_seconds=p.start_seconds,
                                 end_seconds=p.end_seconds)
        ctx.output(result["audio"], "audio")
