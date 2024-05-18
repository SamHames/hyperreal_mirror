"""
Test cases for the index functionality, including integration with some
concrete corpus objects.

"""

import collections
import concurrent.futures as cf
import csv
import logging
import multiprocessing as mp
import pathlib
import random
import shutil
import uuid
from datetime import date

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


@pytest.fixture(name="example_index_corpora_path")
def fixture_example_index_corpora_path(tmp_path):
    "Returns a path to a copy of the example index and corpora in temporary storage."
    random_corpus = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "corpora", "alice.db"), random_corpus)
    random_index = tmp_path / str(uuid.uuid4())
    shutil.copy(pathlib.Path("tests", "index", "alice_index.db"), random_index)

    return random_corpus, random_index


def check_alice():
    """
    Generate test statistics for the Alice corpus against the known tokenisation.

    """
    with open("tests/data/alice30.txt", "r", encoding="utf-8") as f:
        docs = (line[0] for line in csv.reader(f) if line and line[0].strip())
        target_nnz = 0
        target_docs = 0
        target_positions = 0
        for d in docs:
            target_docs += 1
            target_nnz += len(set(hyperreal.utilities.tokens(d)))
            target_positions += sum(
                1 for v in hyperreal.utilities.tokens(d) if v is not None
            )

    return target_docs, target_nnz, target_positions


# This is a list of tuples, corresponding to the corpus class to test, and the
# concrete arguments to that class to instantiate against the test data.
corpora_test_cases = [
    (
        hyperreal.corpus.PlainTextSqliteCorpus,
        [pathlib.Path("tests", "corpora", "alice.db")],
        {},
        check_alice,
    )
]


@pytest.mark.parametrize("corpus,args,kwargs,check_stats", corpora_test_cases)
def test_indexing(pool, tmp_path, corpus, args, kwargs, check_stats):
    """Test that all builtin corpora can be successfully indexed and queried."""
    c = corpus(*args, **kwargs)
    idx = hyperreal.index.Index(tmp_path / corpus.CORPUS_TYPE, c, pool=pool)

    # These are actually very bad settings, but necessary for checking
    # all code paths and concurrency.
    idx.rebuild(
        doc_batch_size=10,
    )

    # Compare against the actual test data.
    target_docs, target_nnz, target_positions = check_stats()

    nnz = list(idx.db.execute("select sum(docs_count) from inverted_index"))[0][0]
    total_docs = list(idx.db.execute("select count(*) from doc_key"))[0][0]
    assert total_docs == target_docs
    assert nnz == target_nnz

    # Feature ids should remain the same across indexing runs
    features_field_values = {
        feature_id: (field, value)
        for feature_id, field, value in idx.db.execute(
            "select feature_id, field, value from inverted_index"
        )
    }

    idx.rebuild(doc_batch_size=10, index_positions=True)

    for feature_id, field, value in idx.db.execute(
        "select feature_id, field, value from inverted_index"
    ):
        assert (field, value) == features_field_values[feature_id]

    idx.rebuild(doc_batch_size=1, index_positions=True)

    positions = list(idx.db.execute("select sum(position_count) from position_index"))[
        0
    ][0]

    assert positions == target_positions

    # Make sure that there's information for every document with
    # positional information.
    assert (
        target_docs
        == list(idx.db.execute("select sum(docs_count) from position_doc_map"))[0][0]
    )

    # Test positional information extraction from documents.
    matching_docs = idx.docs(idx[("text", "hatter")])

    for _, _, doc in matching_docs:
        assert "hatter" in c.doc_to_features(doc)["text"]

    # Test proximity query
    hare_hatter = idx.field_proximity_query("text", [["hare"], ["hatter"]], 50)

    assert len(hare_hatter)
    assert len(hare_hatter) == len(idx[("text", "hare")] & idx[("text", "hatter")])


@pytest.mark.parametrize("n_clusters", [4, 16, 64])
def test_model_creation(pool, example_index_corpora_path, n_clusters):
    """Test creation of a model (the core numerical component!)."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    idx.initialise_clusters(n_clusters)
    # The defaults will generate dense clustering for 4 clusters, hierarchical for 16, 64
    idx.refine_clusters(iterations=3)

    assert len(idx.cluster_ids) == len(idx.top_cluster_features())
    assert 1 < len(idx.cluster_ids) <= n_clusters

    idx.refine_clusters(iterations=3, group_test_batches=0)
    assert len(idx.cluster_ids) == len(idx.top_cluster_features())
    assert 1 < len(idx.cluster_ids) <= n_clusters

    # Initialising with a field that doesn't exist should create an empty model.
    idx.initialise_clusters(n_clusters, include_fields=["banana"])
    idx.refine_clusters(iterations=3)

    assert len(idx.cluster_ids) == len(idx.top_cluster_features())
    assert 0 == len(idx.cluster_ids)

    # No op cases - empty and single clusters selected.
    assert idx._refine_feature_groups({}) == ({}, set())
    idx.refine_clusters(iterations=10, cluster_ids=[1])


def test_model_editing(example_index_corpora_path, pool):
    """Test editing functionality on an index."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    idx.initialise_clusters(16)

    cluster_ids = idx.cluster_ids

    assert len(cluster_ids) == 16

    all_cluster_features = {
        cluster_id: idx.cluster_features(cluster_id) for cluster_id in cluster_ids
    }

    all_feature_ids = [
        feature[0] for features in all_cluster_features.values() for feature in features
    ]

    assert len(all_feature_ids) == len(set(all_feature_ids))

    # Delete single feature at random
    delete_feature_id = random.choice(all_cluster_features[0])[0]
    idx.delete_features([delete_feature_id])
    assert delete_feature_id not in [feature[0] for feature in idx.cluster_features(0)]
    # Deleting the same feature shouldn't fail
    idx.delete_features([delete_feature_id])

    # Delete all features in a cluster
    idx.delete_features([feature[0] for feature in all_cluster_features[0]])
    assert len(idx.cluster_features(0)) == 0

    assert 0 not in idx.cluster_ids and len(idx.cluster_ids) == 15

    idx.delete_clusters([1, 2])
    assert not ({1, 2} & set(idx.cluster_ids)) and len(idx.cluster_ids) == 13

    # Merge clusters
    idx.merge_clusters([3, 4])
    assert 4 not in idx.cluster_ids and len(idx.cluster_ids) == 12

    assert len(idx.cluster_features(3)) == len(all_cluster_features[3]) + len(
        all_cluster_features[4]
    )

    # Create a new cluster from a set of features
    new_cluster_id = idx.create_cluster_from_features(
        [feature[0] for feature in all_cluster_features[4]]
    )
    assert new_cluster_id == 16
    assert len(idx.cluster_ids) == 13


def test_model_structured_sampling(example_index_corpora_path, pool):
    """Test that structured sampling produces something."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    idx.initialise_clusters(8, min_docs=5)
    idx.refine_clusters(iterations=5)

    cluster_sample, sample_clusters = idx.structured_doc_sample(docs_per_cluster=2)

    # Should only a specific number of documents sampled - note that this isn't
    # guaranteed when docs_per_cluster is larger than clusters in the dataset.
    assert (
        len(BitMap.union(*cluster_sample.values()))
        == len(BitMap.union(*sample_clusters.values()))
        == 16
    )

    assert sum(len(docs) for docs in sample_clusters.values()) >= 16

    # Selective cluster exporting
    cluster_sample, sample_clusters = idx.structured_doc_sample(
        docs_per_cluster=2, cluster_ids=idx.cluster_ids[:2]
    )

    assert len(cluster_sample) == 2


def test_querying(example_index_corpora_path, pool):
    """Test some simple applications of boolean querying and rendering of results."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], corpus=corpus, pool=pool)

    idx.initialise_clusters(16)

    query = idx[("text", "the")]
    q = len(query)
    assert q
    assert q == len(list(idx.convert_query_to_keys(query)))
    assert q == len(list(idx.docs(query)))
    assert 5 == len(list(idx.docs(idx.sample_bitmap(query, random_sample_size=5))))

    for _, _, doc in idx.docs(query):
        assert "the" in hyperreal.utilities.tokens(doc["text"])

    # Non existent field raises error:
    with pytest.raises(KeyError):
        x = idx[("nonexistent", "field")]
        assert not x

    # Valid field, no matches, empty result set
    assert not idx["text", "asdlfkjsadlkfjsadf"]

    # Confirm that feature_id -> feature mappings in the model are correct
    # And the cluster queries are in fact boolean combinations.
    for cluster_id in idx.cluster_ids:
        cluster_matching, cluster_bs = idx.cluster_query(cluster_id)

        # Used for checking the ranking with bstm
        accumulator = collections.Counter()

        for feature_id, field, value, _ in idx.cluster_features(cluster_id):
            assert idx[feature_id] == idx[(field, value)]
            assert (idx[feature_id] & cluster_matching) == idx[feature_id]
            for doc_id in idx[feature_id]:
                accumulator[doc_id] += 1

        # also test the ranking with bstm - we should retrieve the same number
        # of results by each method. Note that we check against the number of
        # retrieved top_k from bstm because it returns more than the
        # specified number of results in the event of ties.
        top_k = hyperreal.utilities.bstm(cluster_matching, cluster_bs, 5)
        n_check = len(top_k)
        real_sorted = [
            doc_id for doc_id, _ in sorted(accumulator.items(), key=lambda x: x[1])
        ]
        real_top_k = BitMap(real_sorted[-n_check:])

        assert top_k == real_top_k

    # Confirm that feature lookup works in both directions
    feature = idx.lookup_feature(1)
    assert idx[1] == idx[feature]

    feature_id = idx.lookup_feature_id(("text", "the"))
    assert idx[feature_id] == idx[("text", "the")]


def test_pivoting(example_index_corpora_path, pool):
    """Test pivoting by features and by clusters."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    idx.initialise_clusters(16)

    # Test early/late truncation in each direction with large and small
    # features.
    for scoring in ("jaccard",):
        for query in [("text", "the"), ("text", "denied")]:
            pivoted = idx.pivot_clusters_by_query(idx[query], top_k=2, scoring=scoring)
            for _, _, features in pivoted:
                # This feature should be first in the cluster, but the cluster
                # containing it may not always be first.
                if query == features[0][1:3]:
                    break
            else:
                assert False


@pytest.mark.parametrize("n_clusters", [4, 8, 16])
def test_fixed_seed(example_index_corpora_path, pool, n_clusters):
    """
    Test creation of a model (the core numerical component!).

    """
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(
        example_index_corpora_path[1], pool=pool, corpus=corpus, random_seed=10
    )

    idx.initialise_clusters(n_clusters)
    idx.refine_clusters(iterations=1)

    clustering_1 = idx.top_cluster_features()
    idx.refine_clusters(target_clusters=n_clusters + 5, iterations=2)
    refined_clustering_1 = idx.top_cluster_features()

    # Note we need to initialise a new object with the random seed, otherwise
    # as each random operation consumes items from the stream.
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(
        example_index_corpora_path[1], pool=pool, corpus=corpus, random_seed=10
    )

    idx.initialise_clusters(n_clusters)
    idx.refine_clusters(iterations=1)

    clustering_2 = idx.top_cluster_features()
    idx.refine_clusters(target_clusters=n_clusters + 5, iterations=2)
    refined_clustering_2 = idx.top_cluster_features()

    assert clustering_1 == clustering_2
    assert refined_clustering_1 == refined_clustering_2


def test_splitting(example_index_corpora_path, pool):
    """Test splitting and saving splits works correctly"""

    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    n_clusters = 16
    idx.initialise_clusters(n_clusters)

    assert len(idx.cluster_ids) == n_clusters

    for cluster_id in idx.cluster_ids:
        idx.refine_clusters(cluster_ids=[cluster_id], target_clusters=2, iterations=1)

    assert len(idx.cluster_ids) == n_clusters * 2


def test_dissolving(example_index_corpora_path, pool):
    """Test dissolving clusters, reducing the available cluster count."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    n_clusters = 16
    idx.initialise_clusters(n_clusters)
    idx.refine_clusters(iterations=10)

    assert len(idx.cluster_ids) == n_clusters

    idx.refine_clusters(iterations=4, target_clusters=12)
    assert len(idx.cluster_ids) == 12


def test_filling_empty_clusters(example_index_corpora_path, pool):
    """Test expanding the number of clusters by subdividing the largest."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    n_clusters = 8
    idx.initialise_clusters(n_clusters)
    idx.refine_clusters(iterations=3)
    assert len(idx.cluster_ids) == n_clusters

    idx.refine_clusters(iterations=3, target_clusters=12)

    assert len(idx.cluster_ids) == 12


def test_termination(example_index_corpora_path, caplog, pool):
    """Test that the algorithm actually converges for at least this case."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    n_clusters = 8
    idx.initialise_clusters(n_clusters)

    with caplog.at_level(logging.INFO):
        idx.refine_clusters(iterations=100)
        assert len(idx.cluster_ids) == n_clusters

        for record in caplog.records:
            if "Terminating" in record.message:
                break
        else:
            assert False


def test_pinning(example_index_corpora_path, pool):
    """Test expanding the number of clusters by subdividing the largest."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], pool=pool, corpus=corpus)

    n_clusters = 8
    idx.initialise_clusters(n_clusters, min_docs=5)
    idx.refine_clusters(iterations=1)
    assert len(idx.cluster_ids) == n_clusters

    # Get two features from the first cluster to pin.
    pinned_features = [f[0] for f in idx.cluster_features(1, top_k=2)]
    idx.pin_features(feature_ids=pinned_features)

    # Refine and split at the same time to confirm that splitting also doesn't move pinned features
    idx.refine_clusters(iterations=3, target_clusters=12)

    whole_cluster = {f[0] for f in idx.cluster_features(1)}
    for feature_id in pinned_features:
        assert feature_id in whole_cluster

    assert len(idx.cluster_ids) == 12

    # Now pin a whole cluster
    idx.pin_clusters(cluster_ids=[1])
    idx.refine_clusters(iterations=3, target_clusters=16)

    assert whole_cluster == {f[0] for f in idx.cluster_features(1)}
    assert len(idx.cluster_ids) == 16


def test_indexing_utility(example_index_corpora_path, tmp_path):
    """Test the indexing utility function."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])

    temp_index = tmp_path / "tempindex.db"

    key_id_map = {i: i for i in range(1, 100)}

    hyperreal.index._index_docs(corpus, key_id_map, str(temp_index), 1, mp.Lock())


def test_field_intersection(tmp_path, pool):
    """
    Test computational machinery for intersecting fields with queries.

    This functionality is intended to enable things like evaluating time
    series trends such as calculating how much a particular word is used
    each month.

    """
    data_path = pathlib.Path("tests", "data")
    target_corpora_db = tmp_path / "sx_corpus.db"
    target_index_db = tmp_path / "sx_corpus_index.db"

    sx_corpus = hyperreal.corpus.StackExchangeCorpus(str(target_corpora_db))

    sx_corpus.replace_sites_data(data_path / "chess.meta.stackexchange.com.7z")

    sx_idx = hyperreal.index.Index(str(target_index_db), pool=pool, corpus=sx_corpus)
    sx_idx.rebuild()

    queries = {
        "moves": sx_idx[("Post", "moves")],
        "1st June 2020": sx_idx[("CreationDate", date(2020, 6, 1))],
    }

    _, _, intersections = sx_idx.intersect_queries_with_field(queries, "CreationYear")

    assert all(c > 0 for c in intersections["moves"])

    # the '1st June 2020' query should only have nonzero intersection with a
    # single year.
    assert sum(1 for c in intersections["1st June 2020"] if c > 0) == 1


def test_migration_warning(tmp_path):
    """Test that an appropriate warning is raised for old schema versions."""

    corpus_path = pathlib.Path("tests", "corpora", "alice.db")
    corp = hyperreal.corpus.PlainTextSqliteCorpus(str(corpus_path))

    random_index = tmp_path / str(uuid.uuid4())
    shutil.copy(
        pathlib.Path("tests", "index", "alice_index_old_schema.db"), random_index
    )

    with pytest.raises(hyperreal._index_schema.MigrationError):
        hyperreal.index.Index(str(random_index), corpus=corp)
