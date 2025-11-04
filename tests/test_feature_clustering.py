"""
Test cases for the feature clustering algorithm, and related editing functions.

"""

import concurrent.futures as cf
import multiprocessing as mp
import pathlib
import random

import pytest

from hyperreal import corpus, index_core


@pytest.fixture(scope="module")
def example_idx(tmp_path_factory):
    moby_dick = corpus.TextfileParagraphsCorpus(
        pathlib.Path("tests", "data", "source", "moby_dick.txt")
    )

    idx_path = tmp_path_factory.mktemp("test_data") / "moby_idx.db"

    with cf.ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as pool:
        moby_idx = index_core.HyperrealIndex(
            idx_path,
            moby_dick,
            pool=pool,
        )

        moby_idx.rebuild()

        yield moby_idx


def test_plaintext_feature_cluster(example_idx):

    clustering = example_idx.plugins["feature_clusters"]

    # Initialise a clustering with 32 clusters
    random_clustering = clustering.initialise_random_clustering(
        16, min_docs=5, include_fields=["text"]
    )
    clustering.replace_clusters(random_clustering)

    assert len(clustering.cluster_ids) == 16

    # Refine the clustering
    new_clustering = clustering.refine_clustering()
    retrieved_clustering = clustering.clustering(top_k_features=10)

    assert len(retrieved_clustering) == len(new_clustering) == 16
    for features in retrieved_clustering.values():
        assert len(features) == 10

    # Merge clusters:
    merged_cluster_id = clustering.merge_clusters([1, 0])
    assert len(clustering.cluster_ids) == 15
    assert (
        set(clustering.cluster_features(merged_cluster_id))
        == new_clustering[0] | new_clustering[1]
    )

    # Delete clusters
    clustering.delete_clusters([1])
    retrieved_clustering = clustering.clustering(top_k_features=10)
    assert len(retrieved_clustering) == 14
    assert 0 not in retrieved_clustering

    # Split clusters
    split_cluster_ids = clustering.split_cluster_into(10, 3)
    assert len(split_cluster_ids) == 3
    assert len(clustering.cluster_ids) == 16

    # Refine the clustering using passages rather than docs
    clustering.refine_clustering(iterations=3, use_passages=True)

    assert len(clustering.cluster_ids) == 16

    # Dissolve a cluster
    total_features = sum(
        len(clustering.cluster_features(cluster_id))
        for cluster_id in clustering.cluster_ids
    )

    clustering.dissolve_clusters(list(clustering.cluster_ids)[:3])

    assert len(clustering.cluster_ids) == 13
    assert (
        sum(
            len(clustering.cluster_features(cluster_id))
            for cluster_id in clustering.cluster_ids
        )
        == total_features
    )


def test_dense_sparse_clustering(example_idx):
    """
    Test different parameters for the clustering.

    """

    clustering = example_idx.plugins["feature_clusters"]

    for sampling_rate in (0, 1 / 16, 1 / 2):

        # Initialise a clustering with 32 clusters
        random_clustering = clustering.initialise_random_clustering(
            16, min_docs=5, include_fields=["text"]
        )
        clustering.replace_clusters(random_clustering)
        clustering.refine_clustering(iterations=10, sampling_rate=sampling_rate)


def test_repeatable_runs(example_idx):
    """
    Test that runs with the same initialisation of the random state are repeatable.

    """
    example_idx.random_state = random.Random(42)

    clustering = example_idx.plugins["feature_clusters"]
    clustering.delete_clusters(cluster_ids=clustering.cluster_ids)

    random_clustering = clustering.initialise_random_clustering(
        16, min_docs=5, include_fields=["text"]
    )
    clustering.replace_clusters(random_clustering)
    clustering.refine_clustering(iterations=3)

    first_run = clustering.cluster_ids

    example_idx.random_state = random.Random(42)

    clustering.delete_clusters(cluster_ids=clustering.cluster_ids)

    random_clustering = clustering.initialise_random_clustering(
        16, min_docs=5, include_fields=["text"]
    )
    clustering.replace_clusters(random_clustering)
    clustering.refine_clustering(iterations=3)

    second_run = clustering.cluster_ids

    assert first_run == second_run


def test_cluster_rebuild(example_idx):

    clustering = example_idx.plugins["feature_clusters"]

    clustering.delete_clusters(cluster_ids=clustering.cluster_ids)
    random_clustering = clustering.initialise_random_clustering(
        16, min_docs=5, include_fields=["text"]
    )
    clustering.replace_clusters(random_clustering)

    cluster_ids = clustering.cluster_ids

    example_idx.rebuild()
    assert clustering.cluster_ids == cluster_ids
