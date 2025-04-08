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

from .index_core import (
    HyperrealIndex,
    TableFilter,
    random_sample_bitmap,
)
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
        self.feature_clusters = self.idx.plugins["feature_clusters"]


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
        rendered_features = web_rendering.render_feature_stats_as_dl(
            features,
            url_key="url",
            area_stat="relative_doc_count",
            total_doc_count=total_doc_count,
            display_stat="doc_count",
        )

        linkable_fields = self.idx.field_handlers

        docs = []
        value = self.get_argument("v", None, strip=False)

        if value:
            handler = self.idx.field_handlers[field][0]
            feature = (field, handler.from_url(value))
            matching_docs, count, positions = self.idx[feature]

            sample_docs = random_sample_bitmap(matching_docs, 20)
            docs = self.idx.html_docs(sample_docs)

        sub_nav_links = {
            "Indexed Fields": [
                (f, self.reverse_url("field-features", f))
                for f in self.idx.field_handlers
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
        sub_nav_links = {
            "Indexed Fields": [
                (f, self.reverse_url("field-features", f))
                for f in self.idx.field_handlers
            ]
        }
        self.write(
            web_rendering.full_page(
                f"Indexed Feature Overview",
                [],
                sub_nav_links=sub_nav_links,
                sub_nav_label="Indexed Fields",
            ).render()
        )


def render_facets(idx, query, base_url):

    rendered_facets = []

    for title, features, table_filter in idx.facets:
        sorting = table_filter or TableFilter(order_by="hits")

        faceted = idx.facet_features(query, features)

        for f, stats in faceted.items():
            query_string = idx.feature_to_url_query(f)
            stats["url"] = base_url + query_string

        rendered_facets.append(
            h("li")(
                h("div", klass="cluster-header")(
                    h("h2")(title),
                ),
                web_rendering.render_feature_stats_as_dl(
                    sorting(faceted),
                    area_stat="relative_hits",
                    total_doc_count=idx.total_doc_count,
                    display_stat="hits",
                    url_key="url",
                ),
            )
        )

    return h("ul", klass="cluster")(rendered_facets)


class BrowseClusters(HyperrealRequestHandler):
    def get(self):

        top_k = int(self.get_argument("top_k", "20"))
        f = self.get_argument("f", None, strip=False)
        v = self.get_argument("v", None, strip=False)
        c = self.get_argument("c", None)

        matching_docs = None
        area_stat = "relative_doc_count"

        if f is None and c is None:
            cluster_filter = TableFilter(order_by="relative_doc_count")
            cluster_stats = cluster_filter(self.feature_clusters.cluster_ids)
            clustering = self.feature_clusters.clustering(top_k=int(top_k))

        elif f is not None and v is not None:
            feature = self.idx.feature_from_url((f, v))
            matching_docs, count, _ = self.idx[feature]

        elif c is not None:
            cluster_id = int(c)
            matching_docs, count = self.feature_clusters.cluster_docs(cluster_id)

        else:
            raise Exception()

        docs = []
        facets = None
        base_url = self.reverse_url("browse")

        if matching_docs is not None:
            sample_docs = random_sample_bitmap(matching_docs, 20)
            docs = self.idx.html_docs(sample_docs)
            cluster_stats = self.feature_clusters.facet_clusters_by_query(matching_docs)

            cluster_filter = TableFilter(order_by="jaccard_similarity", keep_above=0)
            cluster_stats = cluster_filter.apply_filter(cluster_stats)
            faceted = self.feature_clusters.facet_clustering_by_query(
                matching_docs, cluster_stats.keys()
            )

            feature_filter = TableFilter(
                order_by="jaccard_similarity", keep_above=0, first_k=int(top_k)
            )
            clustering = {
                cluster_id: feature_filter.apply_filter(features)
                for cluster_id, features in faceted.items()
            }

            area_stat = "jaccard_similarity"

            facets = render_facets(self.idx, matching_docs, base_url)

        else:
            facets = render_facets(self.idx, self.idx.all_doc_ids(), base_url)

        # Update the clusters and features to include a url link
        for cluster_id in cluster_stats.keys():
            cluster_query = f"c={cluster_id}"
            cluster_stats[cluster_id]["url"] = base_url + cluster_query

            for f, stats in clustering[cluster_id].items():
                query_string = self.idx.feature_to_url_query(f)
                stats["url"] = base_url + query_string

        rendered = web_rendering.render_feature_clustering(
            clustering,
            cluster_stats,
            self.idx.total_doc_count,
            url_key="url",
            area_stat=area_stat,
        )

        self.write(
            web_rendering.full_page(
                f"Browse Feature Clusters",
                [rendered, facets, docs or None],
            ).render()
        )


class MainHandler(HyperrealRequestHandler):
    def get(self):

        rendered_fields = {}

        for field, stats in self.idx.indexed_field_summary.items():

            row = stats.copy()
            handler = self.idx.field_handlers[field][0]

            row["Mininum Value"] = handler.to_html(row["Mininum Value"])
            row["Maximum Value"] = handler.to_html(row["Maximum Value"])

            key = h("a", href=self.reverse_url("field-features", field))(field)

            rendered_fields[key] = row

        table = web_rendering.render_field_table(rendered_fields)

        self.write(
            web_rendering.full_page(
                f"Overview of Indexed Fields",
                [table],
            ).render()
        )


def make_index_server(hyperreal_idx: HyperrealIndex, base_path=""):
    return tornado.web.Application(
        handlers=[
            tornado.web.url(rf"{base_path}/", MainHandler, name="home"),
            tornado.web.url(
                rf"{base_path}/indexed-field/?",
                IndexedFieldOverview,
                name="field-index",
            ),
            tornado.web.url(
                rf"{base_path}/indexed-field/([^/]+)",
                IndexedField,
                name="field-features",
            ),
            tornado.web.url(rf"{base_path}/browse/?", BrowseClusters, name="browse"),
        ],
        hyperreal_idx=hyperreal_idx,
        autoreload=True,
    )


async def serve_index(hyperreal_index, base_path=""):
    app = make_index_server(hyperreal_index, base_path)
    app.listen(9999)
    await asyncio.Event().wait()
