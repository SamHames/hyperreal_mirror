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
from urllib.parse import parse_qsl, urlencode

import tornado
from pyroaring import BitMap
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
        search_url = kwargs.get("search_url", self.reverse_url("search"))

        return web_rendering.full_page(
            *args,
            **kwargs,
            search_url=search_url,
            search_fields=self.idx.search_fields,
            extra_css=self.extra_css,
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
            stats["feature_url"] = base_url + "?" + query_string

        html_features = {
            self.idx.feature_to_html(feature): stats
            for feature, stats in features.items()
        }

        total_doc_count = self.idx.total_doc_count
        rendered_features = web_rendering.render_feature_group(
            features, display_hits=False
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
                    web_rendering.column_content_with_header(
                        f"Features in field: {field}",
                        rendered_features,
                    ),
                    web_rendering.column_content_with_header(
                        web_rendering.search_results_header(
                            sample_doc_count, matching_doc_count
                        ),
                        await search_results,
                    ),
                ],
                sub_nav_links=sub_nav_links,
                sub_nav_label="Indexed Fields",
            )
        )


class IndexedFieldOverview(HyperrealRequestHandler):
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

        sub_nav_links = {
            "Indexed fields:": [
                (f, self.reverse_url("field-features", f))
                for f in self.idx.field_handlers
            ]
        }

        self.write(
            self.render_page(
                f"Indexed Feature Overview",
                [
                    web_rendering.column_content_with_header(
                        "Summary of Indexed Fields", table
                    )
                ],
                sub_nav_links=sub_nav_links,
                sub_nav_label="Indexed fields",
            )
        )


def render_facets(
    idx, query, base_url, current_query_encode, select_form_id, highlight_features
):
    rendered_facets = []

    for i, (title, features, table_filter) in enumerate(idx.facets):
        sorting = table_filter or TableFilter()

        faceted = sorting(idx.facet_features(query, features))

        for f, stats in faceted.items():
            feature_encode = idx.feature_to_querystring(f)
            stats["feature_url"] = base_url + "?" + urlencode([("f", feature_encode)])
            stats["hit_count_url"] = (
                base_url
                + "?"
                + urlencode([("f", feature_encode), current_query_encode])
            )
            stats["select_form_value"] = feature_encode

        rendered_facets.append(
            h("li", klass="stack")(
                h("h2", klass="group-header")(title),
                web_rendering.render_feature_group(
                    faceted,
                    select_form_id=select_form_id,
                    select_form_prefix=f"facet-{i}",
                    highlight_features=highlight_features,
                ),
            )
        )

    return h("ul", klass="stack")(rendered_facets)


# This is a temporary implementation of a query language based on DNF - it is expected
# that this will change a lot as the full details, including the index form are figured
# out...


def dnf_query_to_query_string(idx, dnf_query):
    components = []

    for clause in dnf_query:
        n_clauses = len(clause)

        for component in clause:
            if isinstance(component, int):
                components.append(("c", str(component)))

            elif isinstance(component, tuple):
                components.append(("f", idx.feature_to_querystring(component)))

            else:
                raise ValueError(f"Unsupported query element: {component}")

        components.append(("g", str(n_clauses)))

    return urlencode(components)


def query_string_to_dnf_query(idx, query_string):
    components = parse_qsl(query_string)

    clauses = []
    current_clause = []
    highlight_features = set()
    highlight_clusters = set()

    for component_type, component in components:
        if component_type == "c":
            c = int(component)
            current_clause.append(c)
            highlight_clusters.add(c)

        elif component_type == "f":
            feature = idx.feature_from_querystring(component)
            current_clause.append(feature)
            highlight_features.add(feature)

        elif component_type == "g":
            assert len(current_clause) == int(component)
            clauses.append(current_clause)
            current_clause = []

        # Note that there is no else clause - other features are possible but ignored
        # in the query, so we can have various other parameters.

    # Implicitly terminate the last group if unterminated.
    if current_clause:
        clauses.append(current_clause)

    return clauses, highlight_features, highlight_clusters


def evaluate_dnf_query(clustering, dnf_query):
    def evaluate_or_clause(clause):
        result = BitMap()

        for component in clause:
            if isinstance(component, int):
                result |= clustering.cluster_docs(component)[0]
            elif isinstance(component, tuple):
                result |= clustering.idx[component][0]
            else:
                raise ValueError(f"Unsupported query element: {component}")
        return result

    result = evaluate_or_clause(dnf_query[0])

    for clause in dnf_query[1:]:
        result &= evaluate_or_clause(clause)

    return result


def render_dnf_query(clustering, dnf_query):
    def render_clause(clause):
        items = []

        for component in clause:
            if isinstance(component, int):
                top_features = [
                    clustering.idx.feature_to_html(f)
                    for f in clustering.cluster_features(component, top_k_features=3)
                ]
                top_features.append("...")
                rendered = h("span", klass="cluster", style="--gap: var(--s-3);")(
                    f"Cluster: {component} (",
                    h("ul", klass="cluster")(
                        h("li")(feature) for feature in top_features
                    ),
                    ")",
                )
            elif isinstance(component, tuple):
                rendered = clustering.idx.feature_to_html(component)
            else:
                raise ValueError(f"Unsupported query element: {component}")

            items.append(rendered)
            items.append(h("span", klass="query-operator")(" OR "))

        return h("ul", klass="cluster bordered")(h("li")(item) for item in items[:-1])

    rendered_clauses = []
    for clause in dnf_query:
        rendered_clauses.append(render_clause(clause))
        rendered_clauses.append(h("div", klass="query-operator")("AND"))

    return h("div", klass="query cluster")(
        h("ul", klass="stack")(h("li")(clause) for clause in rendered_clauses[:-1]),
    )


class BrowseClusters(HyperrealRequestHandler):
    async def get(self):
        top_k_features = int(self.get_argument("top_k_features", "15"))
        top_k_clusters = int(self.get_argument("top_k_clusters", "30"))

        # The original query at the core of this navigation
        query = self.get_argument("query", "", strip=False)

        # All c's and f's passed in become part of a new or clause in the overall DNF
        # query.
        (
            current_query,
            highlight_features,
            highlight_clusters,
        ) = query_string_to_dnf_query(self.idx, query)

        new_clause, new_features, new_clusters = query_string_to_dnf_query(
            self.idx, self.request.query
        )

        highlight_features |= new_features
        highlight_clusters |= new_clusters

        if new_clause:
            # Pull out duplicates of features currently included into the new clause.
            # This enforces a simplify constraint that a feature can only occur once.
            dedup = set(new_clause[0])
            dedup_query_clauses = [
                [f for f in clause if f not in dedup] for clause in current_query
            ]
            current_query = [clause for clause in dedup_query_clauses if clause]
            current_query.append(new_clause[0])

        current_query_encode = (
            "query",
            dnf_query_to_query_string(self.idx, current_query),
        )

        # Keep track of current query selections so we can return to this view via
        # expand etc.
        return_query_items = [current_query_encode]

        current_query_rendered = h("span")("All Documents")
        if current_query:
            current_query_rendered = render_dnf_query(
                self.feature_clusters, current_query
            )

        expand_cluster_list = [int(e) for e in self.get_arguments("expand")]
        expand_clusters = set(expand_cluster_list)
        if expand_cluster_list:
            anchor_cluster = expand_cluster_list[-1]
        else:
            anchor_cluster = None
            for clause in current_query:
                for component in clause:
                    if isinstance(component, int):
                        anchor_cluster = component

        highlight_clusters = {
            item for clause in current_query for item in clause if isinstance(item, int)
        }
        highlight_features = {
            item
            for clause in current_query
            for item in clause
            if isinstance(item, tuple)
        }

        # We want to highlight the features in each cluster for the query results, but
        # not in the cluster display.
        concordance_highlight_features = {
            f
            for cluster_id in highlight_clusters
            for f in self.feature_clusters.cluster_features(cluster_id)
        } | highlight_features

        skip_feature_pivoting = False

        # Allow returning to this query with expansions intact.
        return_query_items.extend([("expand", c) for c in expand_cluster_list])

        # Special case when nothing is selected to pivot by
        if not current_query:
            matching_docs = self.idx.all_doc_ids()
            matching_doc_count = len(matching_docs)

            # Skip the expensive step for this case where we show all the clusters.
            skip_feature_pivoting = True

        else:
            # TODO: what is selected by the current query.
            matching_docs = evaluate_dnf_query(self.feature_clusters, current_query)
            matching_doc_count = len(matching_docs)

        sample_doc_count = 20
        docs = []
        facets = None
        base_url = self.reverse_url("browse")

        search_results, sample_doc_count = self.render_html_sample_docs(
            matching_docs,
            sample_doc_count,
            highlight_features=concordance_highlight_features,
        )

        order_by_stat = "jaccard_similarity"

        if skip_feature_pivoting:
            cluster_stats = self.feature_clusters.cluster_ids
            clustering = self.feature_clusters.clustering(
                top_k_features=int(top_k_features)
            )

            # Override any clusters with expand = True
            for cluster_id in expand_clusters:
                clustering[cluster_id] = self.feature_clusters.cluster_features(
                    cluster_id
                )

            # Inject jaccard_similarity = relative_doc_count for the case when the
            # implicit query matches all documents.
            for c_stats in cluster_stats.values():
                c_stats["hits"] = c_stats["doc_count"]
                c_stats["jaccard_similarity"] = c_stats["relative_doc_count"]

            for features in clustering.values():
                for f_stats in features.values():
                    f_stats["hits"] = f_stats["doc_count"]
                    f_stats["jaccard_similarity"] = f_stats["relative_doc_count"]

            # Override to show all clusters in this case
            top_k_clusters = matched_cluster_count = len(cluster_stats)
        else:
            cluster_stats = self.feature_clusters.facet_clusters_by_query(matching_docs)
            matched_cluster_count = len(
                TableFilter(order_by=order_by_stat, keep_above=0)(cluster_stats)
            )

        cluster_stats = TableFilter(
            order_by=order_by_stat, keep_above=0, first_k=top_k_clusters
        )(cluster_stats)

        if skip_feature_pivoting:
            faceted = clustering

        else:
            faceted = self.feature_clusters.facet_clustering_by_query(
                matching_docs, cluster_stats.keys()
            )
            # Count the number of features that matched at all, regardless of whether
            # they'll be displayed or not.
            matching_filter = TableFilter(order_by=order_by_stat, keep_above=0)
            for cluster_id, features in faceted.items():
                cluster_stats[cluster_id]["matching_feature_count"] = len(
                    matching_filter(features)
                )

        top_feature_filter = TableFilter(
            order_by=order_by_stat, keep_above=0, first_k=top_k_features
        )
        non_zero_feature_filter = TableFilter(order_by=order_by_stat, keep_above=0)

        clustering = {
            cluster_id: (
                top_feature_filter(features)
                if cluster_id not in expand_clusters
                else non_zero_feature_filter(features)
            )
            for cluster_id, features in faceted.items()
        }

        facets = render_facets(
            self.idx,
            matching_docs,
            base_url,
            current_query_encode,
            "edit-model-form",
            highlight_features=highlight_features,
        )

        # Update the clusters and features to include a url link
        for cluster_id in cluster_stats.keys():
            cluster_stats[cluster_id]["feature_url"] = "".join(
                (base_url, "?", urlencode([("c", str(cluster_id))]))
            )

            cluster_stats[cluster_id]["hit_count_url"] = "".join(
                (
                    base_url,
                    "?",
                    urlencode([("c", str(cluster_id)), current_query_encode]),
                )
            )

            if (
                cluster_id not in expand_clusters
                and cluster_stats[cluster_id]["matching_feature_count"] > top_k_features
            ):
                this_return = [*return_query_items, ("expand", cluster_id)]

                return_url = "".join(
                    (self.reverse_url("browse"), "?", urlencode(this_return))
                )
                cluster_stats[cluster_id]["expand_url"] = return_url

            for f, stats in clustering[cluster_id].items():
                feature = self.idx.feature_to_querystring(f)
                feature_encode = ("f", feature)
                encoded = urlencode([feature_encode])
                stats["feature_url"] = "".join((base_url, "?", encoded))
                stats["hit_count_url"] = "".join(
                    (base_url, "?", urlencode([current_query_encode, feature_encode]))
                )
                stats["select_form_value"] = feature

        see_all_clusters_link = None
        if len(cluster_stats) < matched_cluster_count:
            return_query_items.append(("top_k_clusters", matched_cluster_count))
            return_url = "".join(
                (self.reverse_url("browse"), "?", urlencode(return_query_items))
            )
            see_all_clusters_link = h("div")(
                h("a", href=return_url)("+ ", matched_cluster_count, " more clusters")
            )

        rendered = (
            web_rendering.render_feature_clustering(
                clustering,
                cluster_stats,
                select_form_id="edit-model-form",
                footer=see_all_clusters_link,
                highlight_clusters=highlight_clusters,
                highlight_features=highlight_features,
            ),
        )

        edit_form = web_rendering.render_feature_edit_forms(
            self.reverse_url,
            dnf_query_to_query_string(self.idx, current_query),
        )

        cluster_nav = web_rendering.cluster_navigation(
            self.reverse_url("browse"),
            self.feature_clusters.cluster_ids,
            selected=anchor_cluster,
        )

        self.write(
            self.render_page(
                f"Browse Feature Clusters",
                [
                    (
                        web_rendering.column_content_with_header(
                            "Current Query",
                            current_query_rendered,
                            content_klass="flex-no-grow",
                        ),
                        web_rendering.column_content_with_header(
                            web_rendering.search_results_header(
                                sample_doc_count, matching_doc_count
                            ),
                            await search_results,
                        ),
                    ),
                    web_rendering.column_content_with_header(
                        "Similar Clusters", rendered
                    ),
                    web_rendering.column_content_with_header("Matching Facets", facets),
                ],
                body_header=(
                    edit_form,
                    cluster_nav,
                ),
            )
        )


class NewQueryBrowseClusters(HyperrealRequestHandler):
    def get(self):
        """
        Redirect back to browse to start a new one without the current query.

        """

        args = self.request.arguments.copy()

        # reset the query to empty
        args["query"] = [b""]

        query_string = urlencode(
            [(key, val) for key, values in args.items() for val in values]
        )

        self.redirect(self.reverse_url("browse") + "?" + query_string)


class Search(HyperrealRequestHandler):
    def get(self):
        search_features = []

        search_field = self.get_argument("search-field", strip=False)
        search_value = self.get_argument("search-value", strip=False)

        tokeniser = self.idx.search_fields[search_field]
        # TODO: tokenise should be optional? Default should be to use the field
        # value handler on the given field.
        search_tokens = tokeniser(search_value)

        for value in search_tokens:
            search_features.append(
                ("f", self.idx.feature_to_querystring((search_field, value)))
            )

        self.redirect(self.reverse_url("browse") + "?" + urlencode(search_features))


class CreateCluster(HyperrealRequestHandler):
    def post(self):
        """
        Create a new cluster from the given features.

        """

        # Each feature is a bundled url string (double layered, to let us address
        # arbitrary queries as features
        features = [
            self.idx.feature_from_querystring(f) for f in self.get_arguments("f")
        ]

        if features:
            new_cluster_id = self.feature_clusters.create_cluster_from_features(
                features
            )

            self.redirect(
                self.reverse_url("browse") + f"?c={new_cluster_id}",
            )

        else:
            raise ValueError("no features provided")

    get = post


class MergeClusters(HyperrealRequestHandler):
    def post(self):
        """
        Create a new cluster from the given features.

        """

        # Each feature is a bundled url string (double layered, to let us address
        # arbitrary queries as features
        clusters = [int(c) for c in self.get_arguments("c")]

        if clusters:
            merge_cluster_id = self.feature_clusters.merge_clusters(clusters)

            self.redirect(
                self.reverse_url("browse") + f"?c={merge_cluster_id}",
            )

        else:
            raise ValueError("no clusters provided")

    get = post


class SplitClusters(HyperrealRequestHandler):
    def post(self):
        """
        Split each of the given clusters into two new clusters.

        Clusters that don't exist will be ignored.

        """
        cluster_ids = [int(value) for value in self.get_arguments("c")]

        for cluster_id in sorted(cluster_ids):
            self.feature_clusters.split_cluster_into(cluster_id, 2)

        self.redirect(
            self.reverse_url("browse") + f"?c={cluster_ids[0]}",
        )

    get = post


class DeleteClusters(HyperrealRequestHandler):
    def post(self):
        """
        Delete the given cluster/s from the clustering.

        Clusters that don't exist will be ignored.

        """
        cluster_ids = [int(value) for value in self.get_arguments("c")]

        self.feature_clusters.delete_clusters(cluster_ids)

        self.redirect(
            self.reverse_url("browse"),
        )

    get = post


class RefineClusters(HyperrealRequestHandler):
    def post(self):
        """
        Refine the selected clusters for a certain number of iterations.

        If no clusters are provided, the whole clustering will be refined.

        Clusters that don't exist will be ignored.

        """
        cluster_ids = [int(value) for value in self.get_arguments("c")]

        if cluster_ids:
            sampling_rate = 0
        else:
            cluster_ids = self.feature_clusters.cluster_ids
            sampling_rate = None

        self.feature_clusters.refine_clustering(
            cluster_ids, iterations=10, sampling_rate=sampling_rate
        )

        self.redirect(
            self.reverse_url("browse"),
        )

    get = post


def make_index_server(hyperreal_idx: HyperrealIndex, base_path=""):
    return tornado.web.Application(
        handlers=[
            tornado.web.url(rf"{base_path}/", IndexedFieldOverview, name="home"),
            tornado.web.url(
                rf"{base_path}/indexed-field/",
                IndexedFieldOverview,
                name="field-index",
            ),
            tornado.web.url(
                rf"{base_path}/indexed-field/([^/]+)",
                IndexedField,
                name="field-features",
            ),
            tornado.web.url(rf"{base_path}/browse/", BrowseClusters, name="browse"),
            tornado.web.url(rf"{base_path}/search/", Search, name="search"),
            tornado.web.url(
                rf"{base_path}/browse/new",
                NewQueryBrowseClusters,
                name="browse-new-query",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/create/",
                CreateCluster,
                name="create-cluster",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/merge/",
                MergeClusters,
                name="merge-clusters",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/split/",
                SplitClusters,
                name="split-clusters",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/refine/",
                RefineClusters,
                name="refine-clusters",
            ),
            tornado.web.url(
                rf"{base_path}/cluster/delete/",
                DeleteClusters,
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
