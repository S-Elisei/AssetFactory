from typing import Literal

from core.job import BaseParams, Job, Seed
from pydantic import Field


class QwenVoicedesignSpeech(Job):
    class Params(BaseParams):
        text: str = Field(max_length=1000)
        voice_description: str = Field(max_length=1000)
        language: Literal["auto", "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]
        do_sample: bool
        temperature: float = Field(ge=0.05, le=2.0)
        top_k: int = Field(ge=0, le=2048)
        top_p: float = Field(ge=0.0, le=1.0)
        repetition_penalty: float = Field(ge=1.0, le=2.0)
        subtalker_dosample: bool
        subtalker_temperature: float = Field(ge=0.05, le=2.0)
        subtalker_top_k: int = Field(ge=0, le=2048)
        subtalker_top_p: float = Field(ge=0.0, le=1.0)
        max_new_tokens: int = Field(ge=25, le=4096)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("qwentts", text=p.text, voice_description=p.voice_description,
                                 language=p.language, do_sample=p.do_sample, temperature=p.temperature,
                                 top_k=p.top_k, top_p=p.top_p, repetition_penalty=p.repetition_penalty,
                                 subtalker_dosample=p.subtalker_dosample,
                                 subtalker_temperature=p.subtalker_temperature, subtalker_top_k=p.subtalker_top_k,
                                 subtalker_top_p=p.subtalker_top_p, max_new_tokens=p.max_new_tokens, seed=ctx.seed)
        ctx.output(result["audio"])
