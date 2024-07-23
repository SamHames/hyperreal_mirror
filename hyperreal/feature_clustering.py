"""
Algorithms for clustering features defined on an index.

Features are clustered based on common cooccurrence in documents: this is
operationalised through a notion of average degree of a cluster of features, treating
the features and documents as two types of objects in a bipartite graph.

"""

import array
import collections
import concurrent.futures as cf
import contextlib
import heapq
import math
import mmap
import os
import random
import tempfile

from pyroaring import BitMap


def cluster_features(
    idx,
    features,
    n_clusters,
    iterations,
    layer_sizes=None,
    random_seed=None,
    layer_checks=2,
):
    """
    Create a clustering of the given features.

    """
    rand = random.Random(random_seed)

    # Construct equally sized random clusters for the initialisation.
    features = list(features)
    feature_ids = list(range(len(features)))
    rand.shuffle(feature_ids)

    # This is the clustering initialisation
    leaf_features = {
        leaf_id: set(feature_ids[leaf_id::n_clusters]) for leaf_id in range(n_clusters)
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

    # Convert the feature_ids back to features for the return
    return {
        cluster_id: {features[feature_id] for feature_id in cluster_features}
        for cluster_id, cluster_features in leaf_features.items()
    }


def _score_proposed_moves(idx, mmap_file, clustering, check_features, top_k, offsets):
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
            idx.pool.submit(
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


def construct_mmap(idx, features, mmap_path):
    """
    Materialise features into a file suitable for memory mapping.

    This enables firstly a performance optimisation, skipping the SQLite overhead of
    retrieving lots of features through the blob interface. Secondly this will enable
    moving from base features to clustering of arbitrary queries, such as the results
    of other clusterings or manually constructed queries. For now features are just
    a basic type of query, but this will be expanded in future.

    This also enables the clustering algorithm to use the order of this file as an
    arbitrary compact key that can be squeezed into an array without any trouble.

    """

    current_start = 0
    starts = array.array("q", (0 for _ in features))
    ends = array.array("q", (0 for _ in features))

    with idx, open(mmap_path, "wb") as mm:
        for i, feature in enumerate(features):
            docs = idx[feature]
            docs.run_optimize()
            docs.shrink_to_fit()
            docs = docs.serialize()

            size = len(docs)

            mm.write(docs)
            starts[i] = current_start
            current_start += size
            ends[i] = current_start

    return starts, ends
