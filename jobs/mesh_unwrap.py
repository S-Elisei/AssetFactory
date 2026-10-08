from core.job import BaseParams, Files, Job


class MeshUnwrap(Job):
    class Params(BaseParams):
        mesh: Files([".glb"])

    async def run(self, ctx):
        result = await ctx.local("unwrap", mesh=ctx.file("mesh"))
        ctx.output(result["mesh"])
        ctx.output(result["uv_layout"])
