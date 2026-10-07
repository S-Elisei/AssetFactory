from core.job import BaseParams, Input, Job, Seed
from pydantic import Field


class SfxContinue(Job):
    class Params(BaseParams):
        prompt: str = Field(max_length=1000)
        duration: float = Field(ge=1, le=120)
        steps: int = Field(ge=4, le=16)
        seed: Seed

    inputs = {"audio": Input("audio")}

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("stableaudio", prompt=p.prompt, steps=p.steps, seed=ctx.seed, duration=p.duration,
                                 audio=ctx.file("audio"), noise_level=None, start_seconds=None, end_seconds=None)
        ctx.output(result["audio"], "audio")
