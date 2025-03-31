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
- securely handling user-generated content: certain things (such as corpus specific CSS)
  are not meaningfully escaped or validated

"""

import asyncio
import tornado

from tinyhtml import h

from .index_core import HyperrealIndex
from . import web_rendering


class HyperrealRequestHandler(tornado.web.RequestHandler):
    """
    A regular tornado RequestHandler with access shortcuts to the served index.

    Makes available the following shortcuts:

    self.idx: the HyperrealIndex being served
    self.feature_clusters: the feature clusters plugin.

    """

    def initialize(self):
        self.idx = self.application.settings["hyperreal_idx"]
        self.feature_clusters = self.idx.p.feature_clusters


class IndexedField(HyperrealRequestHandler):
    def get(self, field):
        min_docs = int(self.get_argument("min_docs", "10"))

        features = self.idx.field_features(field, min_docs=min_docs)

        for f, stats in features.items():
            base_url = self.reverse_url("field-features", field)
            query_string = self.idx.feature_to_url_query(f)
            stats["url"] = base_url + "?" + query_string

        html_features = {
            self.idx.feature_to_html(feature): stats
            for feature, stats in features.items()
        }

        total_doc_count = self.idx.total_doc_count
        rendered_features = web_rendering.render_features_as_dl(
            features,
            url_key="url",
            bar_stat="doc_count",
            bar_norm=total_doc_count,
            display_stat="doc_count",
        )

        linkable_fields = self.idx.field_handlers

        docs = []
        value = self.get_argument("v", None, strip=False)

        if value:
            handler = self.idx.field_handlers[field][0]
            feature = (field, handler.from_url(value))
            matching_docs, count, positions = self.idx[feature]

            # TODO: random sampling/ordering/pagination?
            docs = self.idx.html_docs(matching_docs[:20])

        sub_nav_links = {
            "Indexed Fields": [
                (f, f"/indexed-field/{f}") for f in self.idx.field_handlers
            ]
        }
        self.write(
            web_rendering.full_page(
                f"Feature summary for field: {field}",
                [rendered_features, web_rendering.list_docs(docs)],
                sub_nav_links=sub_nav_links,
                sub_nav_label="Indexed Fields",
            ).render()
        )


class IndexedFieldOverview(HyperrealRequestHandler):
    def get(self):
        linkable_fields = self.idx.field_handlers

        self.write(
            web_rendering.indexed_field_page(
                self.idx,
                linkable_fields,
                [],
                None,
                {},
            ).render()
        )


class BrowseClusters(HyperrealRequestHandler):
    def get(self):
        pass


class MainHandler(HyperrealRequestHandler):
    def get(self):
        table_fields = [list(row) for row in self.idx.indexed_field_summary.copy()]

        for row in table_fields[1:]:
            # Conver the value to HTML
            field = row[0]
            handler = self.idx.field_handlers[field][0]

            min_value, max_value = row[4:6]

            row[4:6] = handler.to_html(min_value), handler.to_html(max_value)

        self.write(web_rendering.home_page(table_fields).render())


def make_index_server(hyperreal_idx: HyperrealIndex, base_path=""):
    return tornado.web.Application(
        handlers=[
            tornado.web.url(rf"{base_path}/", MainHandler, name="home"),
            tornado.web.url(
                rf"{base_path}/indexed-field/", IndexedFieldOverview, name="field-index"
            ),
            tornado.web.url(
                rf"{base_path}/indexed-field/([^/]+)",
                IndexedField,
                name="field-features",
            ),
            tornado.web.url(rf"{base_path}/browse/", BrowseClusters, name="browse"),
        ],
        hyperreal_idx=hyperreal_idx,
        autoreload=True,
    )


async def serve_index(hyperreal_index, base_path=""):
    app = make_index_server(hyperreal_index, base_path)
    app.listen(9999)
    await asyncio.Event().wait()
