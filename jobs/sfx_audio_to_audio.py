from core.job import BaseParams, Input, Job, Seed
from pydantic import Field


class SfxAudioToAudio(Job):
    class Params(BaseParams):
        prompt: str = Field(max_length=1000)
        noise_level: float = Field(ge=0.05, le=1.0)
        steps: int = Field(ge=4, le=16)
        seed: Seed

    inputs = {"audio": Input("audio")}

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("stableaudio", prompt=p.prompt, steps=p.steps, seed=ctx.seed, duration=None,
                                 audio=ctx.file("audio"), noise_level=p.noise_level, start_seconds=None,
                                 end_seconds=None)
        ctx.output(result["audio"], "audio")
