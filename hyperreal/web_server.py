"""
The web server component.

Note that this is quite rudimentary and is intended as a local view of a system rather
than a fully web hosted offering. Core functionality like authentication and
authorisation are out of scope right now.

"""

import asyncio
import tornado

from .index_core import HyperrealIndex


class HyperrealRequestHandler(tornado.web.RequestHandler):
    """
    A regular tornado RequestHandler with some shortcuts.

    Makes the index being served available on self.index, as this needs to be accessed
    often.

    """

    def initialize(self):
        self.idx = self.application.settings["hyperreal_idx"]


class FeatureListHandler(HyperrealRequestHandler):
    async def get(self, field):

        min_docs = int(self.get_argument("min_docs", "10"))
        features = self.idx.field_features(field, min_docs=min_docs)

        for i, ((field, value), stats) in enumerate(features.items()):
            stat_line = f"{value}, {', '.join(str(s) for s in stats.values())}\n"
            self.write(stat_line)

            if i % 1000 == 0:
                await self.flush()


def make_app(hyperreal_idx: HyperrealIndex):
    return tornado.web.Application(
        handlers=[
            (r"/feature-list/([^/]+)", FeatureListHandler),
        ],
        hyperreal_idx=hyperreal_idx,
    )


async def serve_index(hyperreal_index):
    app = make_app(hyperreal_index)
    app.listen(9999)
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()
