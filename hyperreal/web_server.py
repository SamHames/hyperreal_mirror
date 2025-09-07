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
from urllib.parse import parse_qsl

import tornado
from tinyhtml import h, raw

from . import web_rendering
from .index_core import HyperrealIndex, TableFilter, random_sample_bitmap


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
        self.extra_css = self.idx.corpus.extra_css

    def render_page(self, *args, **kwargs):

        return web_rendering.full_page(
            *args, **kwargs, extra_css=self.extra_css
        ).render()

    def render_html_sample_docs(
        self, matching_docs, sample_doc_count, highlight_features=None
    ):
        """
        Load and render HTML documents in the background pool.

        Returns an asyncio future - this needs to be awaited.

        """

        retrieve_docs = random_sample_bitmap(matching_docs, sample_doc_count)

        future = asyncio.wrap_future(
            self.idx.pool.submit(
                _render_html_worker,
                (self.idx, retrieve_docs, highlight_features),
            )
        )

        return future, len(retrieve_docs)


class IndexedField(HyperrealRequestHandler):
    async def get(self, field):
        min_docs = int(self.get_argument("min_docs", "10"))

        sample_doc_count = 20

        features = self.idx.field_features(field, min_docs=min_docs)

        for f, stats in features.items():
            base_url = self.reverse_url("field-features", field)
            query_string = self.idx.feature_to_querystring(f)
            stats["url"] = base_url + "?" + query_string

        html_features = {
            self.idx.feature_to_html(feature): stats
            for feature, stats in features.items()
        }

        total_doc_count = self.idx.total_doc_count
        rendered_features = web_rendering.render_feature_stats_table(
            features,
            feature_url_key="url",
            heatmap_stat="relative_doc_count",
            count_stat="doc_count",
        )

        linkable_fields = self.idx.field_handlers

        value = self.get_argument("v", None, strip=False)

        matching_doc_count = total_doc_count

        if value:
            handler = self.idx.field_handlers[field][0]
            feature = (field, handler.from_url(value))
            matching_docs, matching_doc_count, positions = self.idx[feature]

            highlight_features = [feature]

        else:
            highlight_features = None
            matching_docs = self.idx.all_doc_ids()

        search_results, sample_doc_count = self.render_html_sample_docs(
            matching_docs, sample_doc_count, highlight_features=highlight_features
        )

        sub_nav_links = {
            "Indexed Fields": [
                (f, self.reverse_url("field-features", f))
                for f in self.idx.field_handlers
            ]
        }

        self.write(
            self.render_page(
                f"Feature summary for field: {field}",
                [
                    rendered_features,
                    web_rendering.list_search_results(
                        await search_results,
                        sample_doc_count=sample_doc_count,
                        matching_doc_count=matching_doc_count,
                    ),
                ],
                column_flex={1: 2},
                sub_nav_links=sub_nav_links,
                sub_nav_label="Indexed Fields",
            )
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
            self.render_page(
                f"Indexed Feature Overview",
                [],
                sub_nav_links=sub_nav_links,
                sub_nav_label="Indexed Fields",
            )
        )


def render_facets(idx, query, base_url):

    rendered_facets = []

    for title, features, table_filter in idx.facets:
        sorting = table_filter or TableFilter()

        faceted = sorting(idx.facet_features(query, features))

        for f, stats in faceted.items():
            query_string = idx.feature_to_querystring(f)
            stats["url"] = base_url + query_string

        rendered_facets.append(
            h("li")(
                h("div", klass="header")(
                    h("h2")(title),
                ),
                web_rendering.render_feature_stats_table(
                    faceted,
                    heatmap_stat="jaccard_similarity",
                    count_stat="hits",
                    feature_url_key="url",
                ),
            )
        )

    return h("ul", klass="cluster feature-clustering")(rendered_facets)


class BrowseClusters(HyperrealRequestHandler):
    async def get(self):

        top_k_features = int(self.get_argument("top_k_features", "10"))
        top_k_clusters = int(self.get_argument("top_k_clusters", "40"))
        f = self.get_argument("f", None, strip=False)
        v = self.get_argument("v", None, strip=False)
        v1 = self.get_argument("v1", None, strip=False)
        v2 = self.get_argument("v2", None, strip=False)
        c = self.get_argument("c", None)

        heatmap_stat = "jaccard_similarity"
        skip_feature_pivoting = False

        # Special case when nothing is selected to pivot by
        if f is None and c is None:

            matching_docs = self.idx.all_doc_ids()
            matching_doc_count = len(matching_docs)

            highlight_features = None

            # Skip the expensive step for this case where we show all the clusters.
            skip_feature_pivoting = True

            heatmap_stat = "relative_doc_count"

        elif f is not None and v is not None:
            feature = self.idx.feature_from_url((f, v))
            matching_docs, matching_doc_count, _ = self.idx[feature]

            highlight_features = [feature]

        elif f is not None and (v1 or v2 is not None):
            feature = self.idx.feature_from_url((f, v1, v2))
            matching_docs, matching_doc_count, _ = self.idx[feature]

            highlight_features = [feature]

        elif c is not None:
            cluster_id = int(c)
            matching_docs, matching_doc_count = self.feature_clusters.cluster_docs(
                cluster_id
            )

            highlight_features = list(
                self.feature_clusters.cluster_features(cluster_id)
            )

        else:
            raise ValueError("Invalid combination of feature or clusters.")

        sample_doc_count = 20
        docs = []
        facets = None
        base_url = self.reverse_url("browse")

        search_results, sample_doc_count = self.render_html_sample_docs(
            matching_docs, sample_doc_count, highlight_features=highlight_features
        )

        if skip_feature_pivoting:
            cluster_stats = self.feature_clusters.cluster_ids
            clustering = self.feature_clusters.clustering(
                top_k_features=int(top_k_features)
            )
            # Override to show all clusters in this case
            top_k_clusters = len(cluster_stats)
        else:
            cluster_stats = self.feature_clusters.facet_clusters_by_query(matching_docs)

        cluster_filter = TableFilter(
            order_by=heatmap_stat, keep_above=0, first_k=top_k_clusters
        )
        cluster_stats = cluster_filter.apply_filter(cluster_stats)

        if skip_feature_pivoting:
            faceted = clustering
        else:
            faceted = self.feature_clusters.facet_clustering_by_query(
                matching_docs, cluster_stats.keys()
            )

        feature_filter = TableFilter(
            order_by=heatmap_stat, keep_above=0, first_k=top_k_features
        )

        clustering = {
            cluster_id: feature_filter.apply_filter(features)
            for cluster_id, features in faceted.items()
        }

        facets = render_facets(self.idx, matching_docs, base_url)

        # Update the clusters and features to include a url link
        for cluster_id in cluster_stats.keys():
            cluster_query = f"c={cluster_id}"
            cluster_stats[cluster_id]["header_url"] = base_url + cluster_query

            cluster_stats[cluster_id]["seemore_url"] = self.reverse_url(
                "cluster-drilldown", cluster_id
            )

            for f, stats in clustering[cluster_id].items():
                query_string = self.idx.feature_to_querystring(f)
                stats["url"] = base_url + query_string

        rendered = web_rendering.render_feature_clustering(
            clustering,
            cluster_stats,
            feature_url_key="url",
            header_url_key="header_url",
            seemore_url_key="seemore_url",
            heatmap_stat=heatmap_stat,
            count_stat="doc_count",
        )

        self.write(
            self.render_page(
                f"Browse Feature Clusters",
                [
                    rendered,
                    facets,
                    web_rendering.list_search_results(
                        await search_results,
                        sample_doc_count=sample_doc_count,
                        matching_doc_count=matching_doc_count,
                    ),
                ],
            )
        )


class ClusterDrillDownRedirector(HyperrealRequestHandler):
    """
    When no cluster is selected, redirect to the one with the lowest cluster_id.

    """

    def get(self):
        redirect_to_cluster = min(self.feature_clusters.cluster_ids)
        self.redirect(
            self.reverse_url("cluster-drilldown", redirect_to_cluster),
        )


class ClusterDrillDown(HyperrealRequestHandler):
    """
    View for drilling down into a specific cluster, laid out against other clusters.

    """

    async def get(self, cluster_id):

        drill_cluster_id = int(cluster_id)

        top_k_clusters = int(self.get_argument("top_k_clusters", "40"))
        top_k_features = int(self.get_argument("top_k_features", "10"))
        f = self.get_argument("f", None, strip=False)
        v = self.get_argument("v", None, strip=False)
        v1 = self.get_argument("v1", None, strip=False)
        v2 = self.get_argument("v2", None, strip=False)
        c = self.get_argument("c", None)

        matching_docs, _ = self.feature_clusters.cluster_docs(drill_cluster_id)

        other_docs = None

        highlight_features = []
        highlight_clusters = []

        if f is None and c is None:
            pass

        elif f is not None and v is not None:
            feature = self.idx.feature_from_url((f, v))
            other_docs, count, _ = self.idx[feature]

            highlight_features.append(feature)

        elif f is not None and (v1 or v2 is not None):
            feature = self.idx.feature_from_url((f, v1, v2))
            other_docs, count, _ = self.idx[feature]
            highlight_features.append(feature)

        elif c is not None:
            cluster_id = int(c)
            other_docs, count = self.feature_clusters.cluster_docs(cluster_id)
            highlight_clusters.append(cluster_id)

        else:
            raise ValueError("Invalid combination of feature or clusters.")

        if other_docs is not None:
            matching_docs &= other_docs

        matching_doc_count = len(matching_docs)

        docs = []
        facets = None
        base_url = self.reverse_url("cluster-drilldown", drill_cluster_id)
        sample_doc_count = 20

        search_results, sample_doc_count = self.render_html_sample_docs(
            matching_docs, sample_doc_count, highlight_features=highlight_features
        )

        drill_cluster_filter = TableFilter(order_by="jaccard_similarity", keep_above=0)
        cluster_filter = TableFilter(
            order_by="jaccard_similarity", keep_above=0, first_k=top_k_clusters
        )
        feature_filter = TableFilter(
            order_by="jaccard_similarity", keep_above=0, first_k=top_k_features
        )

        # Similarity of matching docs to all clusters
        cluster_similarity = self.feature_clusters.facet_clusters_by_query(
            matching_docs
        )

        # Pull out the selected cluster first before computing the order.
        drill_cluster_order = {drill_cluster_id: cluster_similarity[drill_cluster_id]}

        cluster_order = cluster_filter.apply_filter(cluster_similarity)
        # Ensure that the selected cluster is still there after truncating
        cluster_order[drill_cluster_id] = drill_cluster_order[drill_cluster_id]

        # And then all features for all non-zero similarity clusters
        cluster_feature_order = self.feature_clusters.facet_clustering_by_query(
            matching_docs, cluster_order
        )

        # Pop the currently selected cluster/features out to display in detail
        drill_cluster_features = {
            drill_cluster_id: drill_cluster_filter.apply_filter(
                cluster_feature_order[drill_cluster_id]
            )
        }

        # And remove them from everything else so they can be left in place.
        del cluster_order[drill_cluster_id]
        del cluster_feature_order[drill_cluster_id]

        cluster_feature_order = {
            cluster_id: feature_filter.apply_filter(cluster_feature_order[cluster_id])
            for cluster_id in cluster_order
        }

        facets = render_facets(self.idx, matching_docs, base_url)

        # Update the clusters and features to include a url link
        for cluster_id in cluster_order.keys():
            cluster_query = f"c={cluster_id}"
            cluster_order[cluster_id]["header_url"] = base_url + cluster_query
            cluster_order[cluster_id]["seemore_url"] = self.reverse_url(
                "cluster-drilldown", cluster_id
            )

            for f, stats in cluster_feature_order[cluster_id].items():
                query_string = self.idx.feature_to_querystring(f)
                stats["url"] = base_url + query_string
                stats["feature_form_value"] = query_string

        drill_cluster_order[drill_cluster_id]["header_url"] = base_url
        for f, stats in drill_cluster_features[drill_cluster_id].items():
            query_string = self.idx.feature_to_querystring(f)
            stats["url"] = base_url + query_string
            stats["feature_form_value"] = query_string

        drill_cluster_rendered = web_rendering.render_feature_clustering(
            drill_cluster_features,
            drill_cluster_order,
            self.idx.total_doc_count,
            feature_url_key="url",
            header_url_key="header_url",
            area_stat="jaccard_similarity",
            display_stat="hits",
            feature_form_id="feature-form",
            feature_form_key="feature_form_value",
        )
        other_clusters_rendered = web_rendering.render_feature_clustering(
            cluster_feature_order,
            cluster_order,
            self.idx.total_doc_count,
            feature_url_key="url",
            header_url_key="header_url",
            seemore_url_key="seemore_url",
            area_stat="jaccard_similarity",
            display_stat="hits",
            feature_form_id="feature-form",
            feature_form_key="feature_form_value",
        )

        # If just one feature is selected, use that for highlighting, not the original
        # clustering? This might need some further UI work to make sense of the
        # possible combinations.
        if len(highlight_features) != 1:

            highlight_features.extend(drill_cluster_features[drill_cluster_id])

            for cluster_id in highlight_clusters:
                highlight_features.extend(cluster_feature_order[cluster_id])

        # Link to next, previous clusters, wrapping around to the other end at the limits
        all_clusters = sorted(self.feature_clusters.cluster_ids)
        n_clusters = len(all_clusters)
        drill_loc = all_clusters.index(drill_cluster_id)
        next_cluster = all_clusters[(drill_loc + 1) % n_clusters]
        prev_cluster = all_clusters[(drill_loc - 1) % n_clusters]

        nav_links = {
            "Change Clusters": [
                (
                    "Previous Cluster",
                    self.reverse_url("cluster-drilldown", prev_cluster),
                ),
                ("Next Cluster", self.reverse_url("cluster-drilldown", next_cluster)),
            ]
        }

        form_link = self.reverse_url("create-cluster")
        feature_form = h("form", id="feature-form", method="post", action=form_link)(
            h("button", type="submit")("Create cluster from selected features")
        )

        self.write(
            self.render_page(
                f"Drill down to cluster {drill_cluster_id}",
                [
                    drill_cluster_rendered,
                    other_clusters_rendered,
                    facets,
                    web_rendering.list_search_results(
                        await search_results,
                        sample_doc_count=sample_doc_count,
                        matching_doc_count=matching_doc_count,
                    ),
                ],
                body_header=feature_form,
                column_flex={1: 1.5, 3: 1.5},
                sub_nav_label="Change Clusters",
                sub_nav_links=nav_links,
            )
        )


class CreateCluster(HyperrealRequestHandler):

    def post(self):
        """
        Create a new cluster from the given features.

        """

        # Each feature is a bundled url string (double layered, to let us address
        # arbitrary queries as features
        features = [
            self.idx.feature_from_querystring(f) for f in self.get_arguments("feature")
        ]

        if features:

            new_cluster_id = self.feature_clusters.create_cluster_from_features(
                features
            )

            self.redirect(
                self.reverse_url("cluster-drilldown", new_cluster_id),
            )

        else:
            raise ValueError("no features provided")

    get = post


class DeleteCluster(HyperrealRequestHandler):

    def post(self):
        """
        Delete the given cluster/s from the clustering.

        Clusters that don't exist will be ignored.

        """
        cluster_ids = [int(value) for value in self.get_arguments("c")]

        self.feature_clusters.delete_clusters(cluster_ids)

        redirect_to_cluster = min(self.feature_clusters.cluster_ids)
        self.redirect(
            self.reverse_url("cluster-drilldown", redirect_to_cluster),
        )

    get = post


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
            self.render_page(
                f"Overview of Indexed Fields",
                [table],
            )
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
            tornado.web.url(
                rf"{base_path}/cluster/([0-9]+)/?",
                ClusterDrillDown,
                name="cluster-drilldown",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/?",
                ClusterDrillDownRedirector,
                name="first-cluster",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/create",
                CreateCluster,
                name="create-cluster",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/delete",
                DeleteCluster,
                name="delete-clusters",
            ),
        ],
        hyperreal_idx=hyperreal_idx,
        autoreload=False,
    )


async def serve_index(hyperreal_index, port=9999, base_path=""):
    app = make_index_server(hyperreal_index, base_path)
    app.listen(port)
    await asyncio.Event().wait()


def _render_html_worker(args):

    idx, retrieve_docs, highlight_features = args

    with idx:

        search_results = idx.html_search_results(retrieve_docs, highlight_features)

    return raw(search_results.render())
