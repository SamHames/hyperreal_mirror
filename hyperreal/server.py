"""
Cherrypy based webserver for serving an index (or in future) a set of indexes.

"""

import csv
import io
from collections import defaultdict

import cherrypy
from jinja2 import Environment, PackageLoader, select_autoescape

import hyperreal.index
from hyperreal.corpus import EmptyCorpus

# Cherrypy uses raise HTTPRedirect often, so this lint is just noise.
# pylint: disable=raise-missing-from

# Cherrypy tools are dynamically generated, so there's a lot of noise from here.
# pylint: disable=no-member

# The cherrypy ensure_list tool caused unused argument confusion, as the argument
# needs to be present in the url for the method to process it.
# pylint: disable=unused-argument

# Some of the classes are small but necessary stubs for future expansion.
# pylint: disable=too-few-public-methods

templates = Environment(
    loader=PackageLoader("hyperreal"), autoescape=select_autoescape()
)


@cherrypy.tools.register("before_handler")
def lookup_index():
    """This tool looks up the provided index using the configured webserver."""
    index_id = int(cherrypy.request.params["index_id"])
    cherrypy.request.index = cherrypy.request.config["index_server"].index(index_id)


@cherrypy.tools.register("on_end_request")
def cleanup_index():
    """If an index has been setup for this request, close it."""

    if hasattr(cherrypy.request, "index"):
        cherrypy.request.index.close()


@cherrypy.tools.register("before_handler")
def ensure_list(**kwargs):
    """Ensure that the given variables are always a list of the given type."""

    for key, converter in kwargs.items():
        value = cherrypy.request.params.get(key)
        if value is None:
            cherrypy.request.params[key] = []
        elif isinstance(value, list):
            cherrypy.request.params[key] = [converter(item) for item in value]
        else:
            cherrypy.request.params[key] = [converter(value)]


class Cluster:
    """Endpoints for handling functionality related to a specific cluster of features."""

    @cherrypy.expose
    def index(
        self,
        index_id,
        cluster_id,
        feature_id=None,
        filter_cluster_id=None,
        exemplar_docs="30",
        snippet_window="10",
    ):
        """Display the features and docs from a single cluster of features."""

        # This does need some refactoring work - there's too much complexity here
        # This will require further design work on the querying and display API.
        # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks

        idx = cherrypy.request.index

        cluster_id = int(cluster_id)

        # Redirect to the index overview page to create a new model if no
        # index has been created.
        if not idx.cluster_ids:
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/details")

        ## Cluster specific navigation
        # Work out next/prev clusters for navigation in a ring
        all_clusters = idx.cluster_ids

        try:
            cluster_index = all_clusters.index(cluster_id)
        except ValueError:
            raise cherrypy.HTTPError(404)

        # If we're at the start/end, wrap around to the end/start.
        prev_cluster_id = all_clusters[cluster_index - 1]
        next_cluster_id = all_clusters[
            (cluster_index + 1) if cluster_index < (len(all_clusters) - 1) else 0
        ]

        # Pinned clusters can't be changed, only unpinned
        pinned = int(cluster_id in idx.pinned_cluster_ids)

        # Set defaults to be used if neither cluster/feature_id are provided. In this
        # case there is no query to drive contextualisation, and no docs to display.
        highlight_feature_id = None
        query = idx.cluster_docs(cluster_id)

        html_docs = []
        matches = []
        snippets = []
        snippet_window = int(snippet_window)

        # Assemble a sample of matching documents from the query (if present)
        # A query can be either a single feature_id, or a single cluster_id, or the
        # intersection of a feature and a cluster if both are specified.
        # At the same time assemble the contextual features from the query for
        # highlighting.

        highlight_features = defaultdict(set)

        for feature in idx.cluster_features(cluster_id):
            field, value = feature[1:3]
            highlight_features[field].add(value)

        n_features = sum(len(values) for values in highlight_features.values())

        if feature_id is not None:
            feature_id = highlight_feature_id = int(feature_id)
            query &= idx[feature_id]

            field, value = idx.lookup_feature(feature_id)
            highlight_features[field].add(value)

        if filter_cluster_id is not None:
            filter_cluster_id = int(filter_cluster_id)
            cluster_docs = idx.cluster_docs(filter_cluster_id)
            if query is None:
                query = cluster_docs
            else:
                query &= cluster_docs

            for feature in idx.cluster_features(filter_cluster_id):
                field, value = feature[1:3]
                highlight_features[field].add(value)

        # Sorted features for both display and driving edit navigation
        clusters = list(
            idx.pivot_clusters_by_query(
                query, cluster_ids=[cluster_id], top_k=int(n_features)
            )
        )
        total_docs = len(query)

        sampled_docs = idx.sample_bitmap(query, int(exemplar_docs))

        for _, _, doc in idx.docs(sampled_docs):

            html_docs.append(idx.corpus.doc_to_html(doc))

            doc_features = idx.corpus.doc_to_features(doc)

            feature_matches = idx.match_doc_features(doc_features, highlight_features)

            for field, values in feature_matches.items():
                doc_field_values = doc_features[field]

                # Find matching positions across all values in the field to generate
                # unified concordances

                field_positions = sorted(
                    p for positions in values.values() for p in positions
                )

                if not field_positions:
                    continue

                # Join overlapping segments together into one - this isn't really a
                # true 'concordance', but a blend between a concordance and a
                # snippet.
                starts = [max(0, field_positions[0] - snippet_window)]
                ends = [field_positions[0] + snippet_window]

                for position in field_positions[1:]:
                    start = max(0, position - snippet_window)
                    end = position + snippet_window

                    # If this window overlaps with the previous window
                    if start <= ends[-1]:
                        ends[-1] = end

                    else:
                        starts.append(start)
                        ends.append(end)

                snippets.append(
                    [
                        idx.field_values[field].segment_to_html(
                            doc_field_values,
                            start,
                            end,
                            highlight=values,
                        )
                        for start, end in zip(starts, ends)
                    ]
                )

                # Convert the matching feature to HTML representations as well.
                matches.append(
                    {
                        field: sorted(
                            idx.field_values[field].to_html(value)
                            for value in values.keys()
                        )
                        for field, values in feature_matches.items()
                    }
                )

        fields = idx.field_values.keys()

        # Render all the feature values to HTML
        clusters = [
            (
                cluster_id,
                cluster_score,
                [
                    (feature_id, field, idx.field_values[field].to_html(value), score)
                    for feature_id, field, value, score in features
                ],
            )
            for cluster_id, cluster_score, features in clusters
        ]

        template = templates.get_template("cluster.html")

        return template.generate(
            clusters=clusters,
            total_docs=total_docs,
            search_results=list(zip(html_docs, matches, snippets)),
            # Design note: might be worth letting templates grab the request
            # context, and avoid passing this around for everything that
            # needs it?
            index_id=index_id,
            highlight_feature_id=highlight_feature_id,
            fields=fields,
            context="cluster",
            # Cluster specific nav and editing
            prev_cluster_id=prev_cluster_id,
            next_cluster_id=next_cluster_id,
            pinned=pinned,
            features=clusters[0][-1],
        )

    @cherrypy.expose
    def search(self, index_id, cluster_id, field, value):
        """
        Search a specific field for a specific value.

        Currently this is limited to exact matches on a single value only.

        """
        idx = cherrypy.request.index
        search_value = idx.field_values[field].from_str(value)
        feature_id = idx.lookup_feature_id((field, search_value))

        raise cherrypy.HTTPRedirect(
            f"/index/{index_id}/cluster/{cluster_id}/?feature_id={feature_id}"
        )


@cherrypy.popargs("cluster_id", handler=Cluster())
class ClusterOverview:
    """Endpoints for manipulating clusters, including all or multiple selected clusters."""

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(cluster_id=int)
    def delete(self, index_id, cluster_id=None, return_cluster_id=None, **params):
        """Delete the specified clusters, if they exist."""
        cherrypy.request.index.delete_clusters(cherrypy.request.params["cluster_id"])

        if return_cluster_id:
            redirect_to = f"/index/{index_id}/cluster/{return_cluster_id}"
        else:
            redirect_to = f"/index/{index_id}"
        raise cherrypy.HTTPRedirect(redirect_to)

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(feature_id=int)
    def create(self, index_id, feature_id=None, **params):
        """Create a cluster from the listed features."""
        new_cluster_id = cherrypy.request.index.create_cluster_from_features(
            cherrypy.request.params["feature_id"]
        )
        raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{new_cluster_id}")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(cluster_id=int)
    def merge(self, index_id, cluster_id=None, **params):
        """
        Merge the specified clusters into one.

        The first cluster_id provided is the merged cluster.

        """
        merge_cluster_id = cherrypy.request.index.merge_clusters(
            cherrypy.request.params["cluster_id"]
        )
        raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{merge_cluster_id}")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(cluster_id=int)
    def refine(
        self,
        index_id,
        cluster_id=None,
        target_clusters=None,
        iterations="10",
        return_to="cluster",
        minimum_cluster_features="1",
    ):
        """
        Refine the clustering on this index.

        Optionally provide a list of specific clusters, a number of target clusters
        to expand/contract to, and a minimum number of features in each cluster.

        """
        if target_clusters:
            target_clusters = int(target_clusters)
        else:
            target_clusters = None

        cherrypy.request.index.refine_clusters(
            cluster_ids=cherrypy.request.params["cluster_id"],
            target_clusters=target_clusters,
            minimum_cluster_features=int(minimum_cluster_features),
            iterations=int(iterations),
        )
        if return_to == "cluster":
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{cluster_id[0]}")

        raise cherrypy.HTTPRedirect(f"/index/{index_id}/?cluster_id={cluster_id[0]}")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(cluster_id=int)
    def pin(self, index_id, cluster_id=None, pinned="1", return_to="cluster"):
        """
        Pin or unpin the selected clusters.

        Pinned clusters will not be affected by future iterations of the algorithm.

        """
        cherrypy.request.index.pin_clusters(
            cluster_ids=cherrypy.request.params["cluster_id"], pinned=int(pinned)
        )
        if return_to == "cluster":
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{cluster_id[0]}")

        raise cherrypy.HTTPRedirect(f"/index/{index_id}/?cluster_id={cluster_id[0]}")


class FeatureOverview:
    """Endpoints for specific feature related functionality."""

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(feature_id=int)
    def remove_from_model(self, index_id, feature_id=None, cluster_id=None):
        """Removed the specified features from the model."""
        cherrypy.request.index.delete_features(cherrypy.request.params["feature_id"])
        if cluster_id is not None:
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{cluster_id}")

        raise cherrypy.HTTPRedirect(f"/index/{index_id}/")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(feature_id=int)
    def pin(self, index_id, feature_id=None, cluster_id=None, pinned="1"):
        """
        Pin the selected features.

        Pinned features will not be moved algorithmically, and clusters containing any
        pinned features will not be automatically split.

        """
        cherrypy.request.index.pin_features(
            cherrypy.request.params["feature_id"], pinned=int(pinned)
        )
        if cluster_id is not None:
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/cluster/{cluster_id}")

        raise cherrypy.HTTPRedirect(f"/index/{index_id}/")


@cherrypy.popargs("index_id")
@cherrypy.tools.cleanup_index()
@cherrypy.tools.lookup_index()
class Index:
    """
    Endpoints for functionality across an entire index.

    This includes index summary details, viewing of clustering results and
    searching.

    """

    cluster = ClusterOverview()
    feature = FeatureOverview()

    @cherrypy.expose
    @cherrypy.config(**{"response.stream": True})
    def index(
        self,
        index_id,
        feature_id=None,
        cluster_id=None,
        exemplar_docs="5",
        top_k_features="40",
        snippet_window="10",
    ):
        """Display an entire feature clustering defined on an index."""

        # This does need some refactoring work - there's too much complexity here
        # This will require further design work on the querying and display API.
        # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks

        idx = cherrypy.request.index
        # Redirect to the index overview page to create a new model if no
        # index has been created.
        if not idx.cluster_ids:
            raise cherrypy.HTTPRedirect(f"/index/{index_id}/details")

        # Set defaults to be used if neither cluster/feature_id are provided. In this
        # case there is no query to drive contextualisation, and no docs to display.
        highlight_cluster_id = None
        highlight_feature_id = None
        query = None
        total_docs = 0
        html_docs = []
        matches = []
        snippets = []
        snippet_window = int(snippet_window)

        # Assemble a sample of matching documents from the query (if present)
        # A query can be either a single feature_id, or a single cluster_id, or the
        # intersection of a feature and a cluster if both are specified.
        # At the same time assemble the contextual features from the query for
        # highlighting.

        highlight_features = defaultdict(set)

        if feature_id is not None:
            feature_id = highlight_feature_id = int(feature_id)
            query = idx[feature_id]

            field, value = idx.lookup_feature(feature_id)
            highlight_features[field].add(value)

        if cluster_id is not None:
            cluster_id = highlight_cluster_id = int(cluster_id)
            cluster_docs = idx.cluster_docs(cluster_id)
            if query is None:
                query = cluster_docs
            else:
                query &= cluster_docs

            for feature in idx.cluster_features(cluster_id):
                field, value = feature[1:3]
                highlight_features[field].add(value)

        if query is not None:
            clusters = idx.pivot_clusters_by_query(query, top_k=int(top_k_features))
            total_docs = len(query)

            sampled_docs = idx.sample_bitmap(query, int(exemplar_docs))

            for _, _, doc in idx.docs(sampled_docs):

                html_docs.append(idx.corpus.doc_to_html(doc))

                doc_features = idx.corpus.doc_to_features(doc)

                feature_matches = idx.match_doc_features(
                    doc_features, highlight_features
                )

                for field, values in feature_matches.items():
                    doc_field_values = doc_features[field]

                    # Find matching positions across all values in the field to generate
                    # unified concordances
                    field_positions = sorted(
                        p for positions in values.values() for p in positions
                    )

                    starts = ends = []

                    if field_positions:
                        # Join overlapping segments together into one - this isn't really a
                        # true 'concordance', but a blend between a concordance and a
                        # snippet.
                        starts = [max(0, field_positions[0] - snippet_window)]
                        ends = [field_positions[0] + snippet_window]

                        for position in field_positions[1:]:
                            start = max(0, position - snippet_window)
                            end = position + snippet_window

                            # If this window overlaps with the previous window
                            if start <= ends[-1]:
                                ends[-1] = end

                            else:
                                starts.append(start)
                                ends.append(end)

                    # Convert the matching feature to HTML representations as well.
                    doc_matches = {
                        field: sorted(
                            idx.field_values[field].to_html(value)
                            for value in values.keys()
                        )
                        for field, values in feature_matches.items()
                    }

                    matches.append(doc_matches)

                    doc_snippets = [
                        idx.field_values[field].segment_to_html(
                            doc_field_values,
                            start,
                            end + 1,
                            highlight=values,
                        )
                        for start, end in zip(starts, ends)
                    ]
                    snippets.append(doc_snippets)

        else:
            clusters = idx.top_cluster_features(top_k=int(top_k_features))

        # Render all the feature values to HTML
        clusters = (
            (
                cluster_id,
                cluster_score,
                [
                    (feature_id, field, idx.field_values[field].to_html(value), score)
                    for feature_id, field, value, score in features
                ],
            )
            for cluster_id, cluster_score, features in clusters
        )

        fields = idx.field_values.keys()

        template = templates.get_template("index.html")

        return template.generate(
            clusters=clusters,
            total_docs=total_docs,
            search_results=list(zip(html_docs, matches, snippets)),
            # Design note: might be worth letting templates grab the request
            # context, and avoid passing this around for everything that
            # needs it?
            index_id=index_id,
            highlight_feature_id=highlight_feature_id,
            highlight_cluster_id=highlight_cluster_id,
            fields=fields,
            context="index",
        )

    @cherrypy.expose
    def search(self, index_id, field, value, cluster_id=None):
        """
        Search a specific field for a specific value.

        Currently this is limited to exact matches on a single value only.

        If a cluster_id is provided the search will return to that specific
        cluster view.

        """
        idx = cherrypy.request.index
        search_value = idx.field_values[field].from_str(value)
        feature_id = idx.lookup_feature_id((field, search_value))

        if cluster_id is not None:
            raise cherrypy.HTTPRedirect(
                f"/index/{index_id}/cluster/{cluster_id}?feature_id={feature_id}"
            )

        raise cherrypy.HTTPRedirect(f"/index/{index_id}/?feature_id={feature_id}")

    @cherrypy.expose
    def details(self, index_id):
        """
        Show the details of the index, including indexed fields and associated cardinalities.

        """
        idx = cherrypy.request.index

        template = templates.get_template("details.html")
        current_clusters = len(idx.cluster_ids)
        field_summary = idx.indexed_field_summary()

        return template.render(
            field_summary=field_summary,
            index_id=index_id,
            current_clusters=current_clusters,
        )

    @cherrypy.expose
    def export_clusters(self, index_id):
        """
        Export a spreadsheet of the model information, including features and cluster assignments.

        """

        cherrypy.response.headers["Content-Type"] = "text/csv"
        cherrypy.response.headers["Content-Disposition"] = (
            'attachment; filename="feature_clusters.csv"'
        )
        all_features = cherrypy.request.index.top_cluster_features(top_k=2**62)

        output = io.StringIO()
        writer = csv.writer(output, dialect="excel", quoting=csv.QUOTE_ALL)
        writer.writerow(("cluster_id", "feature_id", "field", "value", "docs_count"))

        for cluster_id, _, cluster_features in all_features:
            for row in cluster_features:
                writer.writerow([cluster_id, *row])

        output.seek(0)

        return cherrypy.lib.file_generator(output)

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    @cherrypy.tools.ensure_list(include_fields=str)
    def recreate_model(
        self,
        index_id,
        include_fields=None,
        min_docs="10",
        clusters="64",
        iterations="10",
        minimum_cluster_features="1",
    ):
        """
        (Re)Create the model for this index with the given parameters.

        Note that this does not actually run any iterations of refinement.

        """
        idx = cherrypy.request.index
        idx.initialise_clusters(
            n_clusters=int(clusters),
            min_docs=int(min_docs),
            include_fields=include_fields or None,
        )

        idx.refine_clusters(
            iterations=int(iterations),
            minimum_cluster_features=int(minimum_cluster_features),
        )

        raise cherrypy.HTTPRedirect(f"/index/{index_id}")

    @cherrypy.expose
    @cherrypy.tools.allow(methods=["POST"])
    def refine_model(
        self,
        index_id,
        iterations="10",
        target_clusters="0",
        minimum_cluster_features="1",
    ):
        """
        Refine the existing model for the given number of iterations.

        """
        if target_clusters:
            target_clusters = int(target_clusters)
        else:
            target_clusters = None
        cherrypy.request.index.refine_clusters(
            iterations=int(iterations),
            target_clusters=target_clusters,
            minimum_cluster_features=int(minimum_cluster_features),
        )

        raise cherrypy.HTTPRedirect(f"/index/{index_id}")


@cherrypy.popargs("index_id", handler=Index())
class IndexOverview:
    """
    Endpoints for functionality relating to all indexes managed by a server.

    Currently this does not do much except link to the single served index.

    """

    @cherrypy.expose
    def index(self):
        "List all indexes known to this server"
        template = templates.get_template("index_listing.html")
        indices = cherrypy.request.config["index_server"].list_indices()
        return template.render(indices=indices)


class Root:
    """
    There will be more things at the base layer in the future.

    But for now we will only worry about the /index layer and
    associated operations.
    """

    @cherrypy.expose
    def index(self):
        "Stub for future expansion."
        raise cherrypy.HTTPRedirect("/index/")


class SingleIndexServer:
    """
    An adapter for serving a single specified index and corpus through the web interface.

    The chosen index always has an index_id of 0.

    """

    def __init__(
        self,
        index_path,
        corpus_class=EmptyCorpus,
        corpus_args=None,
        corpus_kwargs=None,
        pool=None,
    ):
        """
        Helper class for serving a single index via the webserver.

        An index will be created on demand when a request requires.

        This will create a single multiprocessing pool to be shared across
        indexes.

        """
        self.corpus_class = corpus_class
        self.corpus_args = corpus_args
        self.corpus_kwargs = corpus_kwargs
        self.index_path = index_path

        self.pool = pool

    def index(self, index_id):
        """Return the hosted index at index_id=0, otherwise raise 404"""
        if index_id != 0:
            raise cherrypy.HTTPError(404)

        if self.corpus_class:
            corpus = self.corpus_class(
                *(self.corpus_args or []), **(self.corpus_kwargs or {})
            )
        else:
            corpus = None

        return hyperreal.index.Index(self.index_path, corpus=corpus, pool=self.pool)

    def list_indices(self):
        """List the only index defined on this server."""
        return {
            0: (
                self.index_path,
                self.corpus_class,
                self.corpus_args,
                self.corpus_kwargs,
            )
        }


def launch_web_server(index_server, auto_reload=False, port=8080):
    """Launch the web server using the given instance of an index server."""

    cherrypy.config.update({"server.socket_port": port})

    if not auto_reload:
        cherrypy.config.update(
            {
                "global": {
                    "engine.autoreload.on": False,
                }
            }
        )

    cherrypy.tree.mount(
        Root(),
        "/",
        {
            "/": {
                "tools.response_headers.on": True,
                "tools.response_headers.headers": [
                    ("Connection", "close"),
                ],
            }
        },
    )
    cherrypy.tree.mount(
        IndexOverview(),
        "/index",
        {
            "/": {
                "index_server": index_server,
                "tools.response_headers.on": True,
                "tools.response_headers.headers": [
                    ("Connection", "close"),
                ],
            }
        },
    )

    cherrypy.log.access_log.propagate = False
    cherrypy.log.error_log.propagate = False

    cherrypy.engine.signals.subscribe()
    cherrypy.engine.start()
    # Make sure to actually wait for everything to be started.
    cherrypy.engine.wait(cherrypy.engine.states.STARTED)

    return cherrypy.engine
