"""
Cluster features on a given index, to provide a navigable index to a corpus.

"""

from __future__ import annotations

import array
import collections
import concurrent.futures as cf
import contextlib
import heapq
import itertools
import math
import mmap
import multiprocessing as mp
import os
import tempfile
import typing
from functools import cached_property
from random import Random

from pyroaring import BitMap

from .db_utilities import atomic
from .index_plugin import IndexPlugin, Migration
from .query import Query

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
                    """
                    UPDATE cluster set 
                        feature_count = ?, 
                        doc_count = ?, 
                        doc_ids = ? 
                    where cluster_id = ?
                    """,
                    [len(features), len(cluster_docs), cluster_docs, cluster_id],
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
        rows = self.idx.db.execute(
            """
            WITH cluster_details as (
                select 
                    cluster_id,
                    sum(doc_count) as hits
                from feature_cluster
                group by cluster_id
            )
            SELECT
                cluster_id,
                doc_count,
                feature_count,
                hits
            from cluster
            inner join cluster_details using(cluster_id)
            """
        )
        total_docs = self.idx.total_doc_count
        return {
            cluster_id: {
                "doc_count": doc_count,
                "relative_doc_count": doc_count / total_docs,
                "feature_count": feature_count,
                "matching_feature_count": feature_count,
                "hits": hits,
            }
            for cluster_id, doc_count, feature_count, hits in rows
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

        self._move_features_into_cluster(cluster_id, features)

    def _move_features_into_cluster(self, cluster_id, features):
        """Move features into cluster_id."""

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
            ((*self.idx.feature_to_index(f), cluster_id) for f in sorted(features)),
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

        self._replace_features_for_cluster(new_cluster_id, sorted(features))

        return new_cluster_id

    @atomic
    def replace_clusters(self, clustering: Clustering):
        """
        Replace the given clusters on the index.

        If the given clusters already exist they will be removed first. Features
        assigned to other clusters will be moved. Features present on the initial
        clustering but not the replacement will be deleted.

        """

        for cluster_id, features in sorted(clustering.items()):
            self._replace_features_for_cluster(cluster_id, sorted(features))

    @atomic
    def delete_clusters(self, cluster_ids: typing.Iterable[int]) -> None:
        """
        Delete the given clusters from the model (if present).

        """

        self.idx.db.executemany(
            "delete from cluster where cluster_id = ?",
            [(cluster_id,) for cluster_id in cluster_ids],
        )

    def _propose_split_using_current_clusters(self, features, exclude_clusters=None):
        """
        Propose a split of the set of features into new clusters.

        This assigns each feature to a new cluster, based on the objective change from
        moving it to any of the existing clusters (other than in exclude_clusters).

        This can be used to generate new clusters from existing ones, or to assign new
        features to existing clusters.

        """
        exclude_clusters = exclude_clusters or set()

        features = list(features)
        next_best_clusters = [(-1, -1) for _ in features]

        for alternative_cluster_id, cluster_stats in self.cluster_ids.items():
            if alternative_cluster_id in exclude_clusters:
                continue

            cluster_obj = cluster_stats["hits"] / (
                cluster_stats["feature_count"] + cluster_stats["doc_count"]
            )

            cluster_docs, _ = self.cluster_docs(alternative_cluster_id)

            for i, feature in enumerate(features):
                feature_docs, f_docs, _ = self.idx[feature]

                new_cluster_obj = (cluster_stats["hits"] + f_docs) / (
                    cluster_stats["feature_count"]
                    + 1
                    + cluster_docs.union_cardinality(feature_docs)
                )

                next_best_clusters[i] = max(
                    next_best_clusters[i],
                    (new_cluster_obj - cluster_obj, alternative_cluster_id),
                )

        subclusters = collections.defaultdict(set)
        for feature, (_, next_best) in zip(features, next_best_clusters):
            subclusters[next_best].add(feature)

        return subclusters

    @atomic
    def dissolve_clusters(self, cluster_ids):
        """
        Dissolve the given clusters.

        All features from each cluster will be moved to their next best location from
        the remaining clusters.

        """

        dissolve_features = {
            f for cluster_id in cluster_ids for f in self.cluster_features(cluster_id)
        }

        dissolve_to = self._propose_split_using_current_clusters(
            dissolve_features, exclude_clusters=set(cluster_ids)
        )

        for cluster_id, features in dissolve_to.items():
            self._move_features_into_cluster(cluster_id, features)

    @atomic
    def split_cluster_into(self, cluster_id, split_into):
        """
        Split the given cluster into a finer grouping, using the current clustering.

        The initial split will be generated by assigning each feature to it's next best
        cluster, then taking the largest components by number of features as the splits.

        """
        # TODO: this should ideally have an option to take a bulk set of cluster_ids,
        # otherwise splitting a group of clusters will depend on the order of cluster_ids.

        features = set(self.cluster_features(cluster_id))

        # Generate initial split from closest clusters for each.
        initial_split = self._propose_split_using_current_clusters(
            features, set([cluster_id])
        )

        by_size = sorted(
            (
                (len(split), tentative_id)
                for tentative_id, split in initial_split.items()
            ),
            reverse=True,
        )

        # Split off one less than split_into and leave the rest of the cluster in place
        cluster_ids = [cluster_id]

        for _, split_cluster_id in by_size[: split_into - 1]:
            new_cluster_id = self.create_cluster_from_features(
                initial_split[split_cluster_id]
            )
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

        return merge_cluster_id

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
                "doc_count": doc_count,
                "relative_doc_count": doc_count / total_docs,
            }
            for field, value, doc_count in features
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

        self.idx.random_state.shuffle(valid_features)

        clusters = {i: set(valid_features[i::n_clusters]) for i in range(n_clusters)}

        return clusters

    def _generate_mmap_from_clustering(self, clustering, mmap_working_file):

        # Materialise all the features in this cluster in the mmap file
        feature_bitmaps = (
            (f, self.idx[f][0])
            for cluster_features in clustering.values()
            for f in cluster_features
        )

        feature_order, offsets = _mmap_bitmaps(feature_bitmaps, mmap_working_file)

        # generate the surrogate version of the clustering where feature_ids are
        # replaced with just integers and references into the mmap file
        feature_map = {f: i for i, f in enumerate(feature_order)}
        surrogate_clustering = {
            cluster_id: {feature_map[f] for f in cluster_features}
            for cluster_id, cluster_features in clustering.items()
        }

        return feature_order, offsets, surrogate_clustering

    def _split_refine_clustering(
        self,
        clustering,
        n_split_clusters,
        iterations,
        sampling_rate=None,
    ):

        n_clusters = len(clustering)

        # Recursively subdivide the largest clusters until we reach the target.
        cluster_order = []
        for c_id, features in clustering.items():
            heapq.heappush(cluster_order, (-len(features), c_id, features))

        next_cluster_id = max(clustering) + 1

        while len(cluster_order) < n_split_clusters:

            _, c_id, features = heapq.heappop(cluster_order)

            new_features = {f for f in features if self.idx.random_state.random() > 0.5}
            old_features = set(features) - new_features

            heapq.heappush(
                cluster_order, (-len(new_features), next_cluster_id, new_features)
            )
            heapq.heappush(cluster_order, (-len(old_features), c_id, old_features))

            next_cluster_id += 1

        split_clustering = {c_id: features for _, c_id, features in cluster_order}

        # Refine the subdivided clustering
        split_clustering = self._refine_clustering(
            split_clustering, iterations=iterations
        )

        # Then cluster the subdivided clusters
        cluster_order, cluster_clustering = self._cluster_clustering(
            n_clusters, split_clustering, iterations
        )

        return split_clustering, cluster_order, cluster_clustering

    def _cluster_clustering(
        self, n_clusters, clustering, iterations, sampling_rate=None
    ):

        # Construct the mmap of index features for faster loading/saving
        with tempfile.TemporaryDirectory() as tmpdir:
            working_file = os.path.join(tmpdir, "mmap.temp")

            # Materialise all the features in this cluster in the mmap file
            cluster_bitmaps = (
                (c_id, self.idx.match_any(fs)) for c_id, fs in clustering.items()
            )

            cluster_order, offsets = _mmap_bitmaps(cluster_bitmaps, working_file)

            randomised_order = list(range(len(clustering)))
            self.idx.random_state.shuffle(randomised_order)
            surrogate_clustering = {
                c_id: set(randomised_order[c_id::n_clusters])
                for c_id in range(n_clusters)
            }

            surrogate_clustering = _refine_clustering(
                working_file,
                self.idx.pool,
                self.idx.max_workers,
                offsets,
                surrogate_clustering,
                iterations,
                self.idx.random_state,
                sampling_rate,
            )

        return cluster_order, surrogate_clustering

    def _refine_clustering(
        self,
        clustering,
        iterations,
        sampling_rate=None,
    ):

        # Construct the mmap of index features for faster loading/saving
        with tempfile.TemporaryDirectory() as tmpdir, mp.Manager() as manager:
            working_file = os.path.join(tmpdir, "mmap.temp")

            feature_order, offsets, surrogate_clustering = (
                self._generate_mmap_from_clustering(clustering, working_file)
            )

            surrogate_clustering = _refine_clustering(
                working_file,
                self.idx.pool,
                self.idx.max_workers,
                offsets,
                surrogate_clustering,
                iterations,
                self.idx.random_state,
                sampling_rate,
            )

        return {
            cluster_id: {feature_order[i] for i in surrogate_features}
            for cluster_id, surrogate_features in surrogate_clustering.items()
        }

    @atomic
    def refine_clustering(
        self,
        cluster_ids: typing.Optional[typing.Iterable[int]] = None,
        iterations: int = 10,
        sampling_rate: typing.Optional[float] = None,
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
            sampling_rate=sampling_rate,
        )

        self.replace_clusters(refined_clustering)

        return refined_clustering


def _refine_clustering(
    mmap_working_file,
    pool,
    max_workers,
    offsets,
    clustering,
    iterations,
    random_state=None,
    sampling_rate=None,
):
    """
    Refine the given clustering of bitsets given in an mmapped file.

    """
    random_state = random_state or random.Random()

    # Backward mapping of features to clusters to make updates easier.
    f_id_to_cluster = {
        f_id: cluster_id for cluster_id, f_ids in clustering.items() for f_id in f_ids
    }

    total_features = len(offsets[0])
    n_features = len(f_id_to_cluster)
    n_clusters = len(clustering)

    # will be used to randomise the order of feature checks for moving between
    # clusters.
    feature_check_order = array.array("q", f_id_to_cluster)

    with mp.Manager() as manager:

        work_queue = manager.Queue()
        results_queue = manager.Queue()

        # Start the background workers.
        workers = set()

        for _ in range(max_workers):
            workers.add(
                pool.submit(
                    _measure_feature_contribution_to_cluster_worker,
                    mmap_working_file,
                    work_queue,
                    results_queue,
                    offsets,
                )
            )

        best_clusters = []

        # Set a default sampling rate if not specified
        if sampling_rate is None:
            sampling_rate = 1 / (n_clusters**0.5)
        elif sampling_rate <= 0 or sampling_rate > 1:
            raise ValueError(
                f"{sampling_rate=} must be None, or 0 < sampling_rate <= 1"
            )

        # Convert the provided sampling_rate into the number of features to select.
        random_feature_checks = math.ceil(sampling_rate * n_features)

        for iteration in range(iterations):

            best_checked_f_ids = collections.defaultdict(set)

            # Always check against the best cluster from the last iteration -
            # not all moves happen and if we found something better but didn't
            # move we should check it again.
            for f_id, cluster_id in enumerate(best_clusters):
                best_checked_f_ids[cluster_id].add(f_id)

            # Create a permutation to sample features from by indexing.
            # This is an optimisation as otherwise sampling features can be a bottleneck
            # for small collections. Compared to the earlier approach to permutation
            # sampling, this ensures we cover more features on every iteration.
            # This permutation is also the order we check feature moves later.
            random_state.shuffle(feature_check_order)
            sample_start = 0

            # Extend the array to simplify wraparound sampling logic - this way we can
            # always safely just sample from sample_start: sample_start +
            # random_feature_checks
            sample_order = (
                feature_check_order + feature_check_order[:random_feature_checks]
            )

            for cluster_id, f_ids in clustering.items():

                sample_end = sample_start + random_feature_checks

                sample_check_feature_ids = set(
                    feature_check_order[sample_start:sample_end]
                )

                sample_start = sample_end % n_features

                check_feature_ids = (
                    sample_check_feature_ids | best_checked_f_ids[cluster_id]
                ) - f_ids

                # Dispatch work as soon as it's ready!
                work_queue.put(
                    (
                        cluster_id,
                        array.array("q", f_ids),
                        array.array("q", check_feature_ids),
                    )
                )

            # Accumulate the results from the background worker
            objective_estimate = 0
            waiting_for = n_clusters * 3

            best_clusters = array.array("q", (-1 for _ in range(total_features)))
            best_scores = array.array("d", (-math.inf for _ in range(total_features)))

            while waiting_for:
                result = results_queue.get()

                # Error case - we need to exit this and try to terminate the
                # workers to raise the correct error
                if result == "exception":
                    break

                cluster_id = result[0]

                if len(result) == 2:
                    objective_estimate += result[1]

                elif len(result) == 3:
                    for f_id, delta in zip(result[1], result[2]):
                        if f_id not in f_id_to_cluster:
                            continue
                        # Break ties deterministically towards the highest
                        # cluster_id.
                        if (delta, cluster_id) > (
                            best_scores[f_id],
                            best_clusters[f_id],
                        ):
                            best_scores[f_id] = delta
                            best_clusters[f_id] = cluster_id

                waiting_for -= 1

            # Break out of iterations as well so we can observe the worker error.
            if result == "exception":
                break

            possible_moves, applied_moves = _apply_moves(
                best_clusters,
                feature_check_order,
                clustering,
                f_id_to_cluster,
                random_state,
            )

            print(iteration, objective_estimate, possible_moves, applied_moves)

            if possible_moves == 0:
                break

        # Shut down the workers
        for _ in range(len(workers)):
            work_queue.put(None)

        for future in cf.as_completed(workers):
            future.result()

    # Convert the feature_ids back to features for the return
    return clustering


def _mmap_bitmaps(feature_bitmaps, mmap_path):
    """
    Materialise the given iterator of features and bitmaps into a file to open via mmap.

    Returns: list of features in index order, and the start and end offsets of each
        feature.

    """

    current_start = 0
    features = []
    starts = []
    ends = []

    with open(mmap_path, "wb") as mm:

        for feature, bitmap in feature_bitmaps:

            features.append(feature)

            bitmap.run_optimize()
            bitmap.shrink_to_fit()
            bitmap = bitmap.serialize()

            size = len(bitmap)

            mm.write(bitmap)

            starts.append(current_start)
            current_start += size
            ends.append(current_start)

    starts = array.array("q", starts)
    ends = array.array("q", ends)

    return features, (starts, ends)


def _apply_moves(
    best_clusters, check_feature_order, clustering, feature_clusters, rand
):
    """
    Given a set of possible moves, apply the best moves to the given clustering.

    This modifies the provided clustering and feature_clusters in place.

    """

    moves = 0
    possible_moves = 0

    for feature_id in check_feature_order:
        best_cluster = best_clusters[feature_id]
        current_cluster = feature_clusters[feature_id]

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
        prob_acceptance = (from_cluster_size - 1) / total_features

        if prob_acceptance < rand.random():
            continue

        clustering[current_cluster].discard(feature_id)
        clustering[best_cluster].add(feature_id)
        feature_clusters[feature_id] = best_cluster
        moves += 1

    return possible_moves, moves


def _measure_feature_contribution_to_cluster_worker(
    mmap_file, work_queue, results_queue, offsets
):
    """
    Measure objective changes if we move a single feature from one cluster to another.

    This calculates the objective as if we were only moving that single feature. Of
    course we're going to be greedy and try to move them all at the same time (with a
    little stochasticity).

    """

    starts, ends = offsets

    with open(mmap_file, "r+b") as f, contextlib.closing(
        mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    ) as mm:
        try:
            for cluster_key, clustering, check_features in iter(work_queue.get, None):
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

                results_queue.put((cluster_key, objective))

                # PHASE 2: compute the incremental change in objective from removing
                # each feature (alone) from the current cluster.
                return_features = array.array("q", (0 for _ in range(n_features)))
                return_scores = array.array("d", (0 for _ in range(n_features)))

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

                results_queue.put((cluster_key, return_features, return_scores))

                # PHASE 3: Incremental delta from adding new features to the cluster.
                n_check_features = len(check_features)
                return_features = array.array("q", (0 for _ in range(n_check_features)))
                return_scores = array.array("d", (0 for _ in range(n_check_features)))

                i = 0
                # All tokens that are adds (not already in the cluster)
                for feature in check_features:
                    docs = BitMap.deserialize(mm[starts[feature] : ends[feature]])

                    feature_hits = len(docs)

                    new_hits = hits + feature_hits

                    new_c = docs.union_cardinality(cluster_union)

                    # check whether there is any intersection from the union results
                    # using the inclusion-exclusion principle.
                    if c + feature_hits - new_c > 0:
                        new_objective = new_hits / (new_c + (n_features + 1))
                        delta = new_objective - objective

                    # If the feature doesn't intersect with the cluster at all,
                    # give it a bad delta.
                    else:
                        delta = -1

                    return_features[i] = feature
                    return_scores[i] = delta
                    i += 1

                results_queue.put((cluster_key, return_features, return_scores))

        except Exception as e:
            # Mark the exception to the main thread so that it can stop processing
            # and observer the error.
            results_queue.put("exception")
            raise


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
