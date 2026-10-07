from core.job import BaseParams, Input, Job


class MeshUnwrap(Job):
    class Params(BaseParams):
        pass

    inputs = {"mesh": Input("mesh")}

    async def run(self, ctx):
        result = await ctx.local("unwrap", mesh=ctx.file("mesh"))
        ctx.output(result["mesh"], "mesh")
        ctx.output(result["uv_layout"], "image")
