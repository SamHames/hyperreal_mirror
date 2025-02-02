"""
Cluster features on a given index, to provide a navigable index to a corpus.

"""

import dataclasses as dc
import typing

from .index_plugin import Migration, IndexPlugin

if typing.TYPE_CHECKING:
    from .index_core import FieldValues

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
        """
        CREATE table cluster_feature (
            field text not null,
            value,
            value_start,
            value_end,
            cluster_id integer not null,
            doc_count integer default 0,
            primary key (field, value),
            -- Note that this relies on SQLite and other databases handling of unique
            -- with nulls - typically for unique constraints all nulls are considered
            -- distinct so a field, value of ('text', null) is distinct from a separate
            -- ('text', null) value.
            unique(field, value),
            unique(field, value_start, value_end),
            constraint at_least_one_value check(
                coalesce(value, value_start, value_end) is not null
            )
        )
        """,
    ],
    description="Create the main tables for holding a clustering of features.",
)


@dc.dataclass
class Cluster:

    cluster_id: typing.Optional[int] = None
    features: list = dc.field(default_factory=list)
    statistics: dict = dc.field(default_factory=dict)


class FeatureClustering(IndexPlugin):
    """ """

    plugin_name = "feature_clusters"
    current_version = "1"
    migrations = [cluster_migration]

    @property
    def cluster_ids(self):
        """Return the list of all defined clusters."""
        rows = self.idx.db.execute("select cluster_id from cluster")
        return [row[0] for row in rows]

    def post_index(self):
        """Regenerate docs matching each cluster and individual features statistics."""
        # TODO
        pass

    def pivot_clusters_by_query(self, query):
        # TODO: it feels like this should also return either field values or queries?
        # That's beginning to feel like the unifying interface that makes this work...
        # Can we attach some statistics to queries or to field values or something like
        # that. (Can we make field values a base query?).
        pass

    def _construct_mmap(self, queries, mmap_path):
        """
        Materialise queries into a file suitable for memory mapping.

        This enables performance optimisation in three ways:

        - by skipping the SQLite overhead of retrieving features through the blob
          interface: this is not normally an issue for even complex queries, but for
          clustering we access every provided feature on every iteration.
        - by materialising arbitrary queries once at the beginning of the clustering
          process.
        - by representing the provided queries as a compact integer offset, enabling
          intermediate parts of the clustering process to be represented by memory
          efficient arrays.

        """

        current_start = 0
        starts = array.array("q", (0 for _ in features))
        ends = array.array("q", (0 for _ in features))

        with idx, open(mmap_path, "wb") as mm:
            for i, query in enumerate(queries):
                docs = query.evalaute(self.idx)
                docs.run_optimize()
                docs.shrink_to_fit()
                docs = docs.serialize()

                size = len(docs)

                mm.write(docs)

                starts[i] = current_start
                current_start += size
                ends[i] = current_start

        return starts, ends

    # A feature can be in at most one cluster.
    def delete_cluster(self, cluster_id):
        pass

    def set_feature_cluster(self):
        pass

    def save_clustering(self, clustering):
        pass

    def save_cluster(self, cluster_id, cluster):
        pass

    def convert_clustering_to_target_clusters(self, clustering):
        pass

    def refine_clustering(
        self,
        clustering: list[Cluster],
        target_clusters: typing.Optional[int] = None,
        iterations=10,
        layer_sizes=None,
        random_seed=None,
        layer_checks=2,
        feature_fraction_termination_tolerance=0.05,
    ):
        """
        Given a clustering, refine it for the given number of iterations.

        """
        rand = random.Random(random_seed)

        # Construct equally sized random clusters for the initialisation.
        features = list(features)
        feature_ids = list(range(len(features)))
        rand.shuffle(feature_ids)
        n_features = len(feature_ids)

        # This is the clustering initialisation
        leaf_features = {
            leaf_id: set(feature_ids[leaf_id::n_clusters])
            for leaf_id in range(n_clusters)
        }
        # This is the index of features to clusters to make moves easier to compute.
        feature_leaves = {
            feature_id: leaf_id
            for leaf_id, fs in leaf_features.items()
            for feature_id in fs
        }

        leaf_ids = list(leaf_features.keys())

        layer_sizes = layer_sizes or []

        last_layer = n_clusters
        for layer_size in layer_sizes:
            if layer_size / last_layer > 0.5:
                raise ValueError(
                    "Each layer needs to be at most half the size of the last."
                )
            if layer_size <= 2:
                raise ValueError("Layer size should be larger than 2 groups.")
            last_layer = layer_size

        # Construct the mmap of index features for faster loading/saving
        with tempfile.TemporaryDirectory() as tmpdir:
            working_file = os.path.join(tmpdir, "mmap.temp")

            # Defer the construction of the memory map to the background process - this
            # also avoids the awkwardness of the with idx: construction closing the idx
            # passed in...
            future = idx.pool.submit(construct_mmap, idx, features, working_file)
            offsets = future.result()

            all_features = set(feature_leaves)

            for iteration in range(iterations):

                if layer_sizes:

                    # Generate the hierarchy of leaves to check against
                    rand.shuffle(leaf_ids)

                    # We're going to generate surrogate keys for all of the layers
                    # so all references to a cluster are to a single integer.
                    key_maps = {leaf_id: leaf_id for leaf_id in leaf_ids}
                    next_key = max(leaf_features) + 1

                    layer_features = [leaf_features]

                    for layer_size in layer_sizes:
                        next_layer_features = list(layer_features[-1])
                        rand.shuffle(next_layer_features)

                        group_features = {}
                        for i in range(layer_size):
                            group = tuple(next_layer_features[i::layer_size])
                            key_maps[next_key] = group
                            group_features[next_key] = set.union(
                                *(layer_features[-1][subgroup] for subgroup in group)
                            )
                            next_key += 1
                        layer_features.append(group_features)

                    # Dense check against the coarsest layer to initialise
                    check_features = {
                        cluster_key: all_features for cluster_key in layer_features[-1]
                    }

                    for i, layer in enumerate(reversed(layer_features[1:])):
                        objective_estimate, best_move_scores = _score_proposed_moves(
                            idx,
                            working_file,
                            layer,
                            check_features,
                            layer_checks,
                            offsets,
                        )

                        check_features = collections.defaultdict(set)
                        for feature_id, scores in best_move_scores.items():
                            for _, group in scores:
                                for subgroup in key_maps[group]:
                                    check_features[subgroup].add(feature_id)

                else:
                    # At the top level of granularity, check all features against the
                    # combined clusters. Note that at every layer we always check a feature
                    # against it's current cluster, regardless of where the best move is
                    # estimated to be.
                    check_features = {
                        leaf_id: all_features for leaf_id, lf in leaf_features.items()
                    }

                objective_estimate, best_move_scores = _score_proposed_moves(
                    idx, working_file, leaf_features, check_features, 1, offsets
                )

                possible_moves, moves_made = _apply_moves(
                    best_move_scores, feature_ids, leaf_features, feature_leaves, rand
                )

                idx.logger.info(
                    f"{iteration=}, {objective_estimate=}, {possible_moves=}, {moves_made=}"
                )

                if possible_moves / n_features < feature_fraction_termination_tolerance:
                    idx.logger.info(
                        f"Terminating at {iteration=} due to small number of moves."
                    )
                    break

        # Convert the feature_ids back to features for the return
        return {
            cluster_id: {features[feature_id] for feature_id in cluster_features}
            for cluster_id, cluster_features in leaf_features.items()
        }


def _score_proposed_moves(pool, mmap_file, clustering, check_features, top_k, offsets):
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
    best_moves = {
        feature: [(-math.inf, -1)] * top_k
        for cluster_id, features in clustering.items()
        for feature in features
    }

    for future in cf.as_completed(futures):
        result = future.result()
        test_cluster, objective, move_features, move_scores = result
        objective_estimate += objective

        for feature, delta in zip(move_features, move_scores):
            heapq.heappushpop(best_moves[feature], (delta, test_cluster))

    return objective_estimate, best_moves


def _apply_moves(move_scores, check_feature_order, clustering, feature_clusters, rand):
    """
    Given a set of possible moves, apply the best moves to the given clustering.

    This modifies the provided clustering and feature_clusters in place.

    """

    moves = 0
    possible_moves = 0

    for feature in check_feature_order:

        _, best_cluster = max(move_scores[feature])
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
            objective = hits / (c + n_features) - (hits / (hits + n_features))

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
                    old_objective = old_hits / (old_c + (n_features - 1)) - (
                        old_hits / (old_hits + n_features - 1)
                    )

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
                    new_objective = new_hits / (new_c + (n_features + 1)) - (
                        new_hits / (new_hits + n_features + 1)
                    )

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
