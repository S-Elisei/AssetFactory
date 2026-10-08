from typing import Literal

from core.job import BaseParams, Files, Job, Seed
from pydantic import Field


class MusicExtend(Job):
    class Params(BaseParams):
        audio: Files([".wav", ".flac", ".ogg", ".mp3"])
        caption: str = Field(max_length=512)
        lyrics: str = Field(max_length=4096)
        extend_seconds: float = Field(ge=5, le=300)
        vocal_language: Literal[
            "unknown", "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi",
            "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro",
            "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh"]
        steps: int = Field(ge=1, le=8)
        shift: float = Field(ge=1.0, le=5.0)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("acestep", caption=p.caption, lyrics=p.lyrics, duration=None, bpm=None,
                                 keyscale="", timesignature="", vocal_language=p.vocal_language, thinking=False,
                                 rewrite_caption=False, lm_temperature=None, steps=p.steps, shift=p.shift,
                                 start_seconds=None, end_seconds=None, extend_seconds=p.extend_seconds,
                                 cover_strength=None, audio=ctx.file("audio"), seed=ctx.seed)
        ctx.output(result["audio"])
