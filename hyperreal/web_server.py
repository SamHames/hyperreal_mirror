"""
The web server component.

The target environment for this server is to be launched from a Jupyter notebook,
through an environment like BinderHub. The functionality is quite rudimentary and is
intended as a local view of a single index rather than a fully web hosted offering.

Out of scope for this component right now are:

- authentication and authorisation
- multi-user/collaboration
- supporting the process of indexing and clustering - for now this requires indexing
  and clustering to be done and the complete index passed to the server

"""

import asyncio
import tornado

from tinyhtml import h

from .index_core import HyperrealIndex
from . import web_rendering


class HyperrealRequestHandler(tornado.web.RequestHandler):
    """
    A regular tornado RequestHandler with a shortcut to access the served index.

    Makes available the following shortcuts:

    self.idx: the HyperrealIndex being served
    self.feature_clusters: the feature clusters plugin.

    """

    def initialize(self):
        self.idx = self.application.settings["hyperreal_idx"]
        self.feature_clusters = self.idx.p.feature_clusters


class IndexedFieldOverview(HyperrealRequestHandler):
    def get(self, field):
        min_docs = int(self.get_argument("min_docs", "10"))
        features = self.idx.field_features(field, min_docs=min_docs)
        linkable_fields = self.idx.field_handlers

        self.write(
            web_rendering.indexed_field_page(field, features, linkable_fields).render()
        )


class ClusterHandler(HyperrealRequestHandler):
    def get(self, cluster_id):
        pass


class MainHandler(HyperrealRequestHandler):
    def get(self):
        table_fields = self.idx.indexed_field_summary
        self.write(web_rendering.home_page(table_fields).render())


def make_app(hyperreal_idx: HyperrealIndex):
    return tornado.web.Application(
        handlers=[
            (r"/", MainHandler),
            (r"/indexed-field/([^/]+)", IndexedFieldOverview),
            (r"/cluster/([0-9]+)", ClusterHandler),
        ],
        hyperreal_idx=hyperreal_idx,
    )


async def serve_index(hyperreal_index):
    app = make_app(hyperreal_index)
    app.listen(9999)
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()
