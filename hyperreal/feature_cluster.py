"""
Cluster features on a given index, to provide a navigable index to a corpus.

"""

from __future__ import annotations

import array
import collections
import concurrent.futures as cf
import contextlib
import itertools
import math
import mmap
import os
from random import Random
import tempfile
import typing
from functools import cached_property

from pyroaring import BitMap

from .query import Query
from .db_utilities import atomic
from .index_plugin import IndexPlugin, Migration

if typing.TYPE_CHECKING:
    from .index_core import Feature, FeatureStatistics


cluster_migration = Migration(
    from_version=None,
    to_version="1",
    steps=[
        """
        CREATE table cluster (
            cluster_id integer primary key,
            cluster_name text default '',
            notes text default '',
            feature_count integer default 0,
            doc_count integer default 0,
            doc_ids roaring_bitmap
        )
        """,
        # For now this doesn't handle range encode features as they make the data model
        # a lot more complex. Possibly range encoded features won't be that useful
        # anyway.
        """
        CREATE table feature_cluster (
            field text not null,
            value,
            cluster_id integer not null references cluster on delete cascade,
            doc_count integer,
            primary key (field, value)
        )
        """,
        """
        CREATE index cluster_feature on feature_cluster(cluster_id, doc_count desc);
        -- Allow efficiently retrieving features for a given cluster
        """,
        """
        CREATE table changed_cluster(
            /*
            Track any clusters that have their saved features changed.

            This is necessary to know which clusters need to be cleaned up, or have
            their docs rematerialised.

            */
            cluster_id integer primary key references cluster on delete cascade
        );
        """,
        """
        CREATE trigger changing_cluster after insert on changed_cluster
        begin
            insert or ignore into cluster(cluster_id) values (new.cluster_id);
        end
        """,
        """
        CREATE trigger insert_changed_clusters before insert on feature_cluster
        begin
            -- Track clusters of inserted features
            insert or ignore into changed_cluster(cluster_id) values (new.cluster_id);
        end;
        """,
        """
        CREATE trigger delete_changed_clusters before delete on feature_cluster
        begin
            -- Track clusters of deleted features
            insert or ignore into changed_cluster(cluster_id)
                values (old.cluster_id);
        end;
        """,
    ],
    description="Create the main tables for holding a clustering of features.",
)

if typing.TYPE_CHECKING:
    Clustering = dict[typing.Hashable, set[Feature] | FeatureStatistics]


class FeatureClustering(IndexPlugin):
    """
    The materialised state and configuration of clusters of features on an index.

    """

    plugin_name = "feature_clusters"
    current_version = "1"
    migrations = [cluster_migration]

    @cached_property
    def random_state(self):
        return Random()

    def post_index_rebuild(self):
        """Regenerate docs matching each cluster and individual feature statistics."""
        self.idx.db.execute(
            """
            INSERT or ignore into changed_cluster
                select cluster_id from cluster
            """,
        )
        # TODO, probably want to do some of this in parallel.
        self._update_changed_clusters()

        # Update feature_cluster statistics
        self.idx.db.execute(
            """
            UPDATE feature_cluster
                set doc_count = (
                    select doc_count
                    from inverted_index ii
                    where (feature_cluster.field, feature_cluster.value) =
                        (ii.field, ii.value)
                )
            where doc_count is null
            """
        )

    def post_transaction(self):
        """Update clusters and feature statistics after any transaction."""
        self._update_changed_clusters()

    def _update_changed_clusters(self):

        changed_clusters = list(
            self.idx.db.execute("select cluster_id from changed_cluster")
        )

        for (cluster_id,) in changed_clusters:
            features = self.cluster_features(cluster_id)

            if features:
                cluster_docs = self.idx.match_any(features)

                self.idx.db.execute(
                    "UPDATE cluster set doc_count = ?, doc_ids = ? where cluster_id = ?",
                    [len(cluster_docs), cluster_docs, cluster_id],
                )
            else:
                # This is a cluster that needs to be cleaned up from the header as well
                self.idx.db.execute(
                    "DELETE from cluster where cluster_id = ?", [cluster_id]
                )

        self.idx.db.execute("DELETE from changed_cluster")

    @property
    def cluster_ids(self) -> dict[int, dict[str, float]]:
        """Return all defined clusters."""
        rows = self.idx.db.execute("select cluster_id, doc_count from cluster")
        total_docs = self.idx.total_doc_count
        return {
            cluster_id: {
                "doc_count": doc_count,
                "relative_doc_count": doc_count / total_docs,
            }
            for cluster_id, doc_count in rows
        }

    def _create_new_empty_cluster(self) -> int:
        """
        Create a new empty cluster, returning the new cluster_id.

        """
        self.idx.db.execute("insert into cluster(cluster_id) values(null)")
        cluster_id = list(self.idx.db.execute("select last_insert_rowid()"))[0][0]

        return cluster_id

    def _replace_features_for_cluster(self, cluster_id, features):

        # Delete the existing features for the cluster
        self.idx.db.execute(
            "DELETE from feature_cluster where cluster_id = ?", [cluster_id]
        )

        # Delete existing features that might be assigned to another cluster.
        self._delete_features(features)

        self.idx.db.executemany(
            """
            INSERT into feature_cluster(field, value, cluster_id, doc_count) 
                values(
                    ?1, 
                    ?2, 
                    ?3,
                    (
                        select doc_count 
                        from inverted_index ii
                        where (ii.field, ii.value) = (?1, ?2)
                    )
                )
            """,
            ((*self.idx.feature_to_index(f), cluster_id) for f in features),
        )

    @atomic
    def create_cluster_from_features(self, features: typing.Iterable[Feature]) -> int:
        """
        Create a new cluster from the given features.

        Features that are present on the clustering will be reassigned to the new
        cluster.

        """

        if not features:
            raise ValueError(
                f"At least one feature needs to be present, got {features}"
            )

        new_cluster_id = self._create_new_empty_cluster()

        self._replace_features_for_cluster(new_cluster_id, features)

        return new_cluster_id

    @atomic
    def replace_clusters(self, clustering: Clustering):
        """
        Replace the given clusters on the index.

        If the given clusters already exist they will be removed first. Features
        assigned to other clusters will be moved. Features present on the initial
        clustering but not the replacement will be deleted.

        """

        for cluster_id, features in clustering.items():
            self._replace_features_for_cluster(cluster_id, features)

    @atomic
    def delete_clusters(self, cluster_ids: typing.Iterable[int]) -> None:
        """
        Delete the given clusters from the model (if present).

        """

        self.idx.db.executemany(
            "delete from cluster where cluster_id = ?",
            [(cluster_id,) for cluster_id in cluster_ids],
        )

    @atomic
    def split_cluster_into(self, cluster_id, split_into):
        """
        Split the given cluster into split_into new clusters.

        One of the split_into segments will retain the existing cluster_id, the rest
        will be new.

        """

        features = list(self.cluster_features(cluster_id))
        self.random_state.shuffle(features)

        splits = [features[i::split_into] for i in range(split_into)]

        cluster_ids = [cluster_id]

        for split in splits[1:]:
            new_cluster_id = self.create_cluster_from_features(split)
            cluster_ids.append(new_cluster_id)

        return cluster_ids

    @atomic
    def merge_clusters(self, cluster_ids: typing.Iterable[int]) -> None:
        """
        Merge the given clusters, combining all features together.

        """
        merge_clusters = list(cluster_ids)
        merge_cluster_id = merge_clusters[0]

        self.idx.db.executemany(
            "UPDATE feature_cluster set cluster_id = ? where cluster_id = ?",
            [(merge_cluster_id, cluster_id) for cluster_id in merge_clusters[1:]],
        )
        self.idx.db.executemany(
            "INSERT or ignore into changed_cluster values(?)",
            [(c,) for c in merge_clusters],
        )

    def _delete_features(self, features):
        self.idx.db.executemany(
            "DELETE from feature_cluster where (field, value) = (?, ?)",
            (self.idx.feature_to_index(f) for f in features),
        )

    @atomic
    def delete_features(self, features: typing.Iterable[Feature]) -> None:
        """Delete specified features from the clustering (if present)."""

        self._delete_features(features)

    def cluster_docs(self, cluster_id: int) -> BitMap:
        """Return the documents matching the given bitmap."""
        return list(
            self.idx.db.execute(
                "SELECT doc_ids, doc_count from cluster where cluster_id = ?",
                [cluster_id],
            )
        )[0]

    def suggested_clustering_fields(self):
        """
        Returns suggested fields for creating a clustering.

        The suggested fields are the fields that are indexed as more than one value
        per document, such as tokens for text and tags for other documents.

        """

        # Suggestion: check an optional attribute on the corpus?

        return {
            field
            for field, (_, _, cardinality) in self.idx.field_handlers.items()
            if cardinality > 1
        }

    def cluster_features(self, cluster_id, top_k_features=None):
        """Return features and associated statistics for the given cluster."""
        top_k_features = top_k_features or 2**32

        features = self.idx.db.execute(
            """
            SELECT
                field, value, doc_count
            from feature_cluster
            where cluster_id = ?
            order by doc_count desc
            limit ?
            """,
            [cluster_id, top_k_features],
        )

        total_docs = self.idx.total_doc_count

        return {
            self.idx.feature_from_index((field, value)): {
                "docs_count": docs_count,
                "relative_doc_count": docs_count / total_docs,
            }
            for field, value, docs_count in features
        }

    @atomic
    def clustering(self, top_k_features=None):
        """Return the current clustering."""
        return {
            cluster_id: self.cluster_features(cluster_id, top_k_features=top_k_features)
            for cluster_id in self.cluster_ids
        }

    @atomic
    def facet_clusters_by_query(self, query_result, cluster_ids=None):
        """Facet the clusters by the given query_result."""

        cluster_scores = self.cluster_ids

        q_len = len(query_result)

        if cluster_ids is not None:
            cluster_scores = {
                cluster_id: cluster_scores[cluster_id] for cluster_id in cluster_ids
            }

        max_workers = self.idx.max_workers
        to_facet = list(cluster_scores.items())
        batches = [
            (self, query_result, to_facet[i::max_workers]) for i in range(max_workers)
        ]

        batch_results = self.idx.pool.map(_facet_cluster_worker, batches)

        facet_results = {}
        for result in batch_results:
            facet_results |= result

        return facet_results

    @atomic
    def facet_clustering_by_query(self, query_result, cluster_ids=None):
        """Facet the features in each cluster by the given query_result."""

        cluster_ids = list(cluster_ids or self.cluster_ids)
        clustering = {}

        max_workers = self.idx.max_workers
        batches = [
            (self, query_result, cluster_ids[i::max_workers])
            for i in range(max_workers)
        ]

        batch_results = self.idx.pool.map(_facet_clustering_worker, batches)

        for result in batch_results:

            clustering |= result

        return clustering

    @atomic
    def initialise_random_clustering(
        self,
        n_clusters: int,
        min_docs: int = 10,
        include_fields=None,
    ):
        """
        Return a randomly initialised clustering for the given number of clusters.

        If include_fields is None, will create initial clusters using
        suggested_clustering_fields.

        """

        if include_fields is not None:
            indexed_fields = set(self.idx.field_handlers)

            if invalid_fields := set(include_fields) - indexed_fields:
                raise ValueError(f"Fields {invalid_fields} do not exist on this index.")

            fields = set(include_fields)

        else:
            fields = self.suggested_clustering_fields()

        valid_features = []

        for field in fields:
            valid_features.extend(self.idx.field_features(field, min_docs=min_docs))

        self.random_state.shuffle(valid_features)

        clusters = {i: set(valid_features[i::n_clusters]) for i in range(n_clusters)}

        return clusters

    def _refine_clustering(
        self,
        clustering: Clustering,
        iterations: int = 10,
        group_test_n_clusters: typing.Optional[int] = None,
        random_group_checks: int = 1,
        moving_feature_fraction_tolerance: float = 0.05,
    ) -> Clustering:

        # Part 1, as preparation we'll convert all the features into integer surrogate
        # keys for memory mapping.
        surrogate_clusters = {}
        features = []
        current_feature_id = 0

        # Generate surrogate identifiers for each feature, and reconstruct the
        # clustering with respect to these surrogates.
        for cluster, cluster_features in clustering.items():
            feature_ids = set()

            for f in cluster_features:
                features.append(f)
                feature_ids.add(current_feature_id)
                current_feature_id += 1

            surrogate_clusters[cluster] = feature_ids

        # Inverse mapping from surrogate id to cluster.
        surrogate_feature_cluster = {
            f_id: cluster_id
            for cluster_id, surrogates in surrogate_clusters.items()
            for f_id in surrogates
        }

        n_features = len(features)
        n_clusters = len(surrogate_clusters)

        # will be used to randomise the order of feature checks for moving between
        # clusters.
        feature_check_order = array.array("q", surrogate_feature_cluster)

        leaf_ids = list(surrogate_clusters.keys())

        if (
            group_test_n_clusters is not None
            and group_test_n_clusters > n_clusters * 0.5
        ):
            raise ValueError(
                f"{group_test_n_clusters=} must be less than half of {n_clusters=} to "
                "have any benefit"
            )

        # Construct the mmap of index features for faster loading/saving
        with tempfile.TemporaryDirectory() as tmpdir:
            working_file = os.path.join(tmpdir, "mmap.temp")

            # Construct the mmap in the background - this might take a little bit.
            future = self.idx.pool.submit(
                _construct_mmap, self.idx, features, working_file
            )
            offsets = future.result()

            all_features = set(surrogate_feature_cluster)

            for _ in range(iterations):

                if group_test_n_clusters is None:

                    # Slow path: check all features against all clusters.
                    check_features = {
                        leaf_id: all_features
                        for leaf_id, lf in surrogate_clusters.items()
                    }

                else:
                    # Fast and approximate path: conduct group tests, checking features
                    # against unions of clusters to avoid checking all features against
                    # all clusters.
                    self.random_state.shuffle(leaf_ids)

                    group_keys = [
                        leaf_ids[i::group_test_n_clusters]
                        for i in range(group_test_n_clusters)
                    ]

                    group_cluster_features = {
                        i: set.union(*(surrogate_clusters[c] for c in keys))
                        for i, keys in enumerate(group_keys)
                    }

                    group_check_features = {
                        i: all_features for i, _ in enumerate(group_keys)
                    }

                    _, best_moves = _score_proposed_moves(
                        self.idx.pool,
                        working_file,
                        group_cluster_features,
                        group_check_features,
                        offsets,
                    )

                    check_features = collections.defaultdict(set)

                    for feature_id, best_cluster_group in enumerate(best_moves):
                        # Test against the best group, plus a random sample of other
                        # groups.
                        random_groups = self.random_state.sample(
                            group_keys, random_group_checks
                        )
                        best_group = group_keys[best_cluster_group]
                        test_clusters = itertools.chain(best_group, *random_groups)

                        for cluster_id in test_clusters:
                            check_features[cluster_id].add(feature_id)

                _, best_clusters = _score_proposed_moves(
                    self.idx.pool,
                    working_file,
                    surrogate_clusters,
                    check_features,
                    offsets,
                )

                self.random_state.shuffle(feature_check_order)
                possible_moves, _ = _apply_moves(
                    best_clusters,
                    feature_check_order,
                    surrogate_clusters,
                    surrogate_feature_cluster,
                    self.random_state,
                )

                if possible_moves / n_features < moving_feature_fraction_tolerance:
                    break

        # Convert the feature_ids back to features for the return
        return {
            cluster_id: {features[feature_id] for feature_id in cluster_features}
            for cluster_id, cluster_features in surrogate_clusters.items()
        }

    @atomic
    def refine_clustering(
        self,
        cluster_ids: typing.Optional[typing.Iterable[int]] = None,
        iterations: int = 10,
        group_test_n_clusters: typing.Optional[int] = None,
        random_group_checks: int = 1,
        moving_feature_fraction_tolerance: float = 0.05,
    ):
        """
        Refine the current clustering defined on the index.

        This is an interface over the refine_clustering function that is specific to the
        materialised representation of a feature clustering on an index. If you'd like
        to experiment more see the _refine_clustering which does not save the resulting
        state.

        """

        cluster_ids = cluster_ids or self.cluster_ids
        clustering = {
            cluster_id: self.cluster_features(cluster_id) for cluster_id in cluster_ids
        }

        refined_clustering = self._refine_clustering(
            clustering,
            iterations=iterations,
            group_test_n_clusters=group_test_n_clusters,
            random_group_checks=random_group_checks,
            moving_feature_fraction_tolerance=moving_feature_fraction_tolerance,
        )

        self.replace_clusters(refined_clustering)

        return refined_clustering


def _construct_mmap(
    idx, features_or_queries: typing.Iterable[Feature | Query], mmap_path
):
    """
    Materialise features or queries into a file suitable for memory mapping.

    This enables performance optimisation in three ways:

    - by skipping the SQLite overhead of retrieving features through the blob
      interface: this is not normally an issue for even complex features, but for
      clustering we access every provided feature on every iteration.
    - by materialising arbitrary features once at the beginning of the clustering
      process.
    - by representing the provided features as a compact integer offset, enabling
      intermediate parts of the clustering process to be represented by memory
      efficient arrays.
    - the resulting array is used as a memory map, meaning we don't need to hold the
      whole set in memory at once, and we only need to keep the page cache for the
      specific sections we need out of the full index.

    """

    current_start = 0
    starts = array.array("q", (0 for _ in features_or_queries))
    ends = array.array("q", (0 for _ in features_or_queries))

    with idx, open(mmap_path, "wb") as mm:
        for i, feature_or_query in enumerate(features_or_queries):

            if isinstance(feature_or_query, Query):
                docs = feature_or_query.evaluate(idx)
            else:
                docs = idx[feature_or_query][0]

            docs.run_optimize()
            docs.shrink_to_fit()
            docs = docs.serialize()

            size = len(docs)

            mm.write(docs)

            starts[i] = current_start
            current_start += size
            ends[i] = current_start

    idx.close()

    return starts, ends


def _score_proposed_moves(
    pool,
    mmap_file,
    clustering,
    check_features,
    offsets,
):
    """Score the proposed sets of moves between clusters."""

    futures = set()

    # dispatch clusters with the most features to process first
    sort_order = sorted(
        clustering.items(),
        reverse=True,
        key=lambda x: len(x[1] | check_features[x[0]]),
    )

    for cluster_id, fs in sort_order:
        futures.add(
            pool.submit(
                _measure_feature_contribution_to_cluster,
                mmap_file,
                cluster_id,
                fs,
                check_features[cluster_id] - fs,
                offsets,
            )
        )

    objective_estimate = 0

    n_features = len(offsets[0])
    best_clusters = array.array("q", (-1 for _ in range(n_features)))
    best_scores = array.array("d", (-math.inf for _ in range(n_features)))

    for future in cf.as_completed(futures):
        result = future.result()
        test_cluster, objective, move_features, move_scores = result
        objective_estimate += objective

        for feature, delta in zip(move_features, move_scores):
            if delta > best_scores[feature]:
                best_scores[feature] = delta
                best_clusters[feature] = test_cluster

    return objective_estimate, best_clusters


def _apply_moves(move_scores, check_feature_order, clustering, feature_clusters, rand):
    """
    Given a set of possible moves, apply the best moves to the given clustering.

    This modifies the provided clustering and feature_clusters in place.

    """

    moves = 0
    possible_moves = 0

    for feature in check_feature_order:

        best_cluster = move_scores[feature]
        current_cluster = feature_clusters[feature]

        # This is not a move
        if best_cluster == current_cluster:
            continue

        possible_moves += 1

        from_cluster_size = len(clustering[current_cluster])
        to_cluster_size = len(clustering[best_cluster])
        total_features = from_cluster_size + to_cluster_size

        # probabilistically control cluster size by adjusting chance of movement
        # based on relative cluster sizes: make moves from clusters with many more
        # features to few features almost certain, and very difficult from clusters
        # with few features to many features. Clusters with similar numbers of
        # features are accepted with 50/50 chance effectively.
        # Note also: the numerator is -1: we calculate the probability according
        # to the size of from cluster after the feature is removed, so that we
        # avoid emptying a cluster out.
        prob_acceptance = ((from_cluster_size - 1) / total_features) ** 2

        if prob_acceptance < rand.random():
            continue

        clustering[current_cluster].discard(feature)
        clustering[best_cluster].add(feature)
        feature_clusters[feature] = best_cluster
        moves += 1

    return possible_moves, moves


def _measure_feature_contribution_to_cluster(
    mmap_file, cluster_key, clustering, check_features, offsets
):
    """
    Measure objective change if we moved a single feature from one cluster to another.

    This calculates the objective as if we were only moving that single feature. Of
    course we're going to be greedy and try to move them all at the same time (with a
    little stochasticity).

    """
    starts, ends = offsets

    with open(mmap_file, "r+b") as f:
        with contextlib.closing(
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        ) as mm:
            # FIRST PHASE: compute the objective and minimal cover stats for the
            # current cluster.

            # The union of all docs covered by the cluster
            cluster_union = BitMap()
            # The set of all docs covered at least twice.
            # This will be used to work out which documents are only covered once.
            covered_twice = BitMap()

            hits = 0
            n_features = len(clustering)

            if n_features == 0:
                return (cluster_key, 0, [], [])

            # Construct the union of all cluster tokens, and also the set of
            # documents only covered by a single feature.
            for feature in clustering:
                docs = BitMap.deserialize(mm[starts[feature] : ends[feature]])

                hits += len(docs)

                # Docs covered at least twice
                covered_twice |= cluster_union & docs
                # All docs now covered
                cluster_union |= docs

            only_once = cluster_union - covered_twice

            c = len(cluster_union)
            objective = hits / (c + n_features)

            # PHASE 2: compute the incremental change in objective from removing
            # each feature (alone) from the current cluster.

            n_return_features = n_features + len(check_features)
            return_features = array.array("q", (0 for _ in range(n_return_features)))
            return_scores = array.array("d", (0 for _ in range(n_return_features)))

            i = 0

            # Features that are already in the cluster, so we need to calculate a remove
            # operator. Effectively we're counting the negative of the score for
            # removing that feature as the effect of adding it to the cluster.
            for feature in clustering:
                docs = BitMap.deserialize(mm[starts[feature] : ends[feature]])

                feature_hits = len(docs)

                old_hits = hits - feature_hits
                only_once_hits = docs.intersection_cardinality(only_once)
                old_c = c - only_once_hits

                # Check if this feature intersects with any other feature in this cluster
                intersects_with_other_feature = only_once_hits < feature_hits

                # It's okay for the cluster to become empty - we'll just prune it.
                if old_c and intersects_with_other_feature:
                    old_objective = old_hits / (old_c + (n_features - 1))

                    delta = objective - old_objective

                # Penalises features that don't intersect with other features in the
                # cluster.
                elif old_c:
                    delta = -1
                # If it would otherwise be a singleton cluster, mark it as a bad move
                else:
                    delta = -1

                return_features[i] = feature
                return_scores[i] = delta
                i += 1

            # PHASE 3: Incremental delta from adding new features to the cluster.

            # All tokens that are adds (not already in the cluster)
            for feature in check_features:
                docs = BitMap.deserialize(mm[starts[feature] : ends[feature]])

                feature_hits = len(docs)

                if docs.intersect(cluster_union):
                    new_hits = hits + feature_hits
                    new_c = docs.union_cardinality(cluster_union)
                    new_objective = new_hits / (new_c + (n_features + 1))

                    delta = new_objective - objective

                # If the feature doesn't intersect with the cluster at all,
                # give it a bad delta.
                else:
                    delta = -1

                return_features[i] = feature
                return_scores[i] = delta
                i += 1

            assert i == n_return_features

    return cluster_key, objective, return_features, return_scores


def _facet_cluster_worker(args):
    fc_plugin, query_result, cluster_stats = args

    q_len = len(query_result)
    results = {}

    with fc_plugin.idx:

        for cluster_id, stats in cluster_stats:
            docs, doc_count = fc_plugin.cluster_docs(cluster_id)

            inter = query_result.intersection_cardinality(docs)
            stats["hits"] = inter

            # Compute some other derived statistics
            stats["jaccard_similarity"] = inter / (doc_count + q_len - inter)
            stats["feature_proportion"] = inter / doc_count
            stats["query_proportion"] = inter / q_len

            results[cluster_id] = stats

    return results


def _facet_clustering_worker(args):

    fc_plugin, query_result, cluster_ids = args

    result = {}

    with fc_plugin.idx:
        for cluster_id in cluster_ids:
            features = fc_plugin.cluster_features(cluster_id)
            result[cluster_id] = fc_plugin.idx.facet_features(query_result, features)

    return result
