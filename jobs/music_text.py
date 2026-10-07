from typing import Literal

from core.job import BaseParams, Job, Seed
from pydantic import Field


class MusicText(Job):
    class Params(BaseParams):
        caption: str = Field(max_length=512)
        lyrics: str = Field(max_length=4096)
        duration: float = Field(ge=5, le=600)
        bpm: int = Field(ge=0, le=300)
        keyscale: Literal[
            "", "C major", "C# major", "Cb major", "D major", "D# major", "Db major", "E major", "E# major",
            "Eb major", "F major", "F# major", "Fb major", "G major", "G# major", "Gb major", "A major", "A# major",
            "Ab major", "B major", "B# major", "Bb major", "C minor", "C# minor", "Cb minor", "D minor", "D# minor",
            "Db minor", "E minor", "E# minor", "Eb minor", "F minor", "F# minor", "Fb minor", "G minor", "G# minor",
            "Gb minor", "A minor", "A# minor", "Ab minor", "B minor", "B# minor", "Bb minor"]
        timesignature: Literal["", "2", "3", "4", "6"]
        vocal_language: Literal[
            "unknown", "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi",
            "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro",
            "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh"]
        thinking: bool
        rewrite_caption: bool
        lm_temperature: float = Field(ge=0.1, le=2.0)
        steps: int = Field(ge=1, le=8)
        shift: float = Field(ge=1.0, le=5.0)
        seed: Seed

    async def run(self, ctx):
        p = ctx.params
        result = await ctx.local("acestep", caption=p.caption, lyrics=p.lyrics, duration=p.duration, bpm=p.bpm,
                                 keyscale=p.keyscale, timesignature=p.timesignature,
                                 vocal_language=p.vocal_language, thinking=p.thinking,
                                 rewrite_caption=p.rewrite_caption, lm_temperature=p.lm_temperature,
                                 steps=p.steps, shift=p.shift, start_seconds=None, end_seconds=None,
                                 extend_seconds=None, cover_strength=None, audio=None, seed=ctx.seed)
        ctx.output(result["audio"], "audio")
