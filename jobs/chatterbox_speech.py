from typing import Literal

from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class ChatterboxSpeech(Job):
    class Params(BaseParams):
        voice_reference: Files([".wav", ".flac", ".ogg", ".mp3"], 0, 1)
        text: str = Field(max_length=600)
        language: Literal["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl",
                          "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"]
        exaggeration: float = Field(ge=0.25, le=2.0)
        cfg_weight: float = Field(ge=0.0, le=1.0)
        temperature: float = Field(ge=0.05, le=2.0)
        repetition_penalty: float = Field(ge=1.0, le=2.0)
        min_p: float = Field(ge=0.0, le=1.0)
        top_p: float = Field(ge=0.0, le=1.0)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("chatterbox", text=p.text, language=p.language, exaggeration=p.exaggeration,
                                 cfg_weight=p.cfg_weight, temperature=p.temperature,
                                 repetition_penalty=p.repetition_penalty, min_p=p.min_p, top_p=p.top_p,
                                 voice_reference=ctx.file("voice_reference"), seed=ctx.seed)
        ctx.output(result["audio"])
