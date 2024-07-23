"""
Test functionality for anything that interacts with saved clusters of features.

"""

import concurrent.futures as cf
import logging
import math
import multiprocessing as mp
import pathlib
import shutil
import uuid

import pytest
from pyroaring import BitMap

import hyperreal


@pytest.fixture(scope="module", name="pool")
def fixture_pool():
    """
    A ProcessPoolExecutor that can be reused for the whole module.

    Avoids spinning up/down a new process pool for every test.

    """
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(4, mp_context=context) as process_pool:
        yield process_pool


@pytest.fixture(name="example_idx")
def fixture_example_idx(tmp_path, pool):
    "Returns a path to a copy of the example index and corpora in temporary storage."
    random_corpus = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "corpora", "alice.db"), random_corpus)
    random_index = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "index", "alice_index.db"), random_index)

    corp = hyperreal.corpus.PlainTextSqliteCorpus(random_corpus)
    idx = hyperreal.index.Index(random_index, corp, pool=pool)

    return idx


def initialise_clusters(idx, n_clusters=10, features_per_half_cluster=10):
    """Given an index, generate some arbitrary clusters."""

    # Drop the document count as we go.
    top_features = [row[0] for row in idx.field_features("text", min_docs_count=5)]

    # Generates 10 clusters with 20 overlapping features.
    for i in range(n_clusters):
        range_start = i * features_per_half_cluster
        range_end = (i + 2) * features_per_half_cluster
        cluster = top_features[range_start:range_end]

        idx.create_cluster_from_features(cluster)


def test_cluster_create_delete_list_annotate(example_idx):
    """
    Test creating, listing and deleting by id clusters of arbitrary query combinations.

    """

    initialise_clusters(example_idx, n_clusters=10, features_per_half_cluster=10)

    for cluster_id in example_idx.cluster_ids:
        new_name = f"Cluster: {cluster_id}"
        note = "This is an unrelated annotation"

        # Defaults are empty
        assert ("", "") == example_idx.cluster_annotations(cluster_id)

        # Updating one should not update the other.
        example_idx.update_cluster_annotations(cluster_id, name=new_name)
        assert (new_name, "") == example_idx.cluster_annotations(cluster_id)
        example_idx.update_cluster_annotations(cluster_id, notes=note)
        assert (new_name, note) == example_idx.cluster_annotations(cluster_id)

    assert example_idx.cluster_ids == list(range(1, 11))

    example_idx.delete_clusters(range(1, 6))

    assert len(example_idx.cluster_ids) == 5

    for cluster_id in example_idx.cluster_ids:
        assert len(example_idx.cluster_features(cluster_id)) == 20

    example_idx.restore_deleted_clusters(range(1, 6))

    assert len(example_idx.cluster_ids) == 10

    for cluster_id in example_idx.cluster_ids:
        assert len(example_idx.cluster_features(cluster_id)) == 20


@pytest.mark.parametrize("n_clusters", [9, 16, 64])
def test_feature_clustering(example_idx, n_clusters):
    """Test different combinations of the automatic feature clustering."""

    features = [row[0] for row in example_idx.field_features("text", min_docs_count=2)]

    prev_clusters = 0

    # we'll test both flat and layered clusterings for each total n_clusters to
    # exercise more of the code paths.
    for layer_sizes in (None, [math.ceil(n_clusters**0.5)]):
        # Note we're fixing the seed so we can recheck
        clustering = hyperreal.cluster_features(
            example_idx,
            features,
            n_clusters,
            10,
            layer_sizes=layer_sizes,
            random_seed=n_clusters,
        )

        assert len(clustering) == n_clusters

        for cluster in clustering.values():
            example_idx.create_cluster_from_features(cluster)

        total_clusters = len(example_idx.cluster_ids)
        assert total_clusters - prev_clusters == n_clusters
        prev_clusters = total_clusters

        # Last check, reproducible enough clustering runs
        reclustering = hyperreal.cluster_features(
            example_idx,
            features,
            n_clusters,
            10,
            layer_sizes=layer_sizes,
            random_seed=n_clusters,
        )

        assert reclustering == clustering


@pytest.mark.parametrize("layer_sizes", [None, [3], [16], [32, 4]])
def test_good_layer_sizes(example_idx, layer_sizes):
    """Test several good layer sizes for clusterings with 64 total clusters."""

    features = [row[0] for row in example_idx.field_features("text", min_docs_count=2)]

    n_clusters = 64

    # we'll test both flat and layered clusterings for each total n_clusters to exercise
    # more of the code paths.
    clustering = hyperreal.cluster_features(
        example_idx, features, n_clusters, 10, layer_sizes=layer_sizes
    )

    assert len(clustering) == n_clusters


@pytest.mark.parametrize("layer_sizes", [[0], [2], [4, 32]])
def test_bad_layer_sizes(example_idx, layer_sizes):
    """Test layersizes that should raise errors."""

    features = [row[0] for row in example_idx.field_features("text", min_docs_count=2)]

    # we'll test both flat and layered clusterings for each total n_clusters to exercise
    # more of the code paths.
    with pytest.raises(ValueError):
        hyperreal.cluster_features(
            example_idx, features, 64, 10, layer_sizes=layer_sizes
        )


def test_cluster_rebuild_after_index_rebuild(tmp_path):
    """
    Test that clusters are correctly rebuilt after the index is rebuilt.

    """

    # TODO

    # Create a corpus with sample data

    # Create clusters on that corpus

    # Recreate the corpus with double the same data and check that the queries are
    # all doubled in length.


def test_model_structured_sampling(example_idx):
    """Test that structured sampling produces something using the current clusters."""

    initialise_clusters(example_idx, n_clusters=10, features_per_half_cluster=10)

    cluster_sample, sample_clusters = example_idx.structured_doc_sample(
        docs_per_cluster=2
    )

    # Should only a specific number of documents sampled - note that this isn't
    # guaranteed when docs_per_cluster is larger than clusters in the dataset.
    assert (
        len(BitMap.union(*cluster_sample.values()))
        == len(BitMap.union(*sample_clusters.values()))
        == 20
    )

    assert sum(len(docs) for docs in sample_clusters.values()) >= 20

    # Selective cluster exporting
    cluster_sample, sample_clusters = example_idx.structured_doc_sample(
        docs_per_cluster=2, cluster_ids=example_idx.cluster_ids[:2]
    )

    assert len(cluster_sample) == 2


def test_pivoting(example_idx):
    """Test pivoting by features and by clusters."""

    features = [row[0] for row in example_idx.field_features("text", min_docs_count=1)]
    clustering = hyperreal.cluster_features(
        example_idx,
        features,
        16,
        iterations=10,
    )

    for cluster in clustering.values():
        example_idx.create_cluster_from_features(cluster)

    # Test early/late truncation in each direction with large and small
    # features.
    for query in [("text", "the"), ("text", "denied")]:
        pivoted = list(example_idx.pivot_clusters_by_query(example_idx[query], top_k=2))
        for _, _, features in pivoted:
            # This feature should be first in the cluster, but the cluster
            # containing it may not always be first.
            if query == features[0][0]:
                break
        else:
            assert False


def test_pivoting_utility(example_idx):
    """Test utility function correctly produces top 10"""

    features = [row[0] for row in example_idx.field_features("text", min_docs_count=1)]
    clustering = hyperreal.cluster_features(
        example_idx,
        features,
        16,
        iterations=10,
    )

    for cluster in clustering.values():
        example_idx.create_cluster_from_features(cluster)

    query = example_idx[("text", "the")]
    inter = query.intersection_cardinality(example_idx.cluster_docs(1))

    top_10_features = {
        feature
        for _, feature in sorted(
            [
                (query.jaccard_index(example_idx[feature]), feature)
                for feature, _ in example_idx.cluster_features(1)
            ],
            reverse=True,
        )[:10]
    }

    # Test the background process
    _, _, features = hyperreal.index._pivot_cluster_features_by_query_jaccard(
        example_idx, query, 1, 10, inter
    )

    computed_top_10 = {feature for feature, _ in features}

    assert computed_top_10 == top_10_features


def test_termination(example_idx, caplog):
    """Test that the algorithm actually converges for at least this case."""

    with caplog.at_level(logging.INFO):
        features = [
            row[0] for row in example_idx.field_features("text", min_docs_count=3)
        ]
        hyperreal.cluster_features(
            example_idx,
            features,
            16,
            iterations=100,
        )

        for record in caplog.records:
            if "Terminating" in record.message:
                break
        else:
            assert False
