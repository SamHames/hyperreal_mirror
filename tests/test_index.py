"""
Test cases for the index functionality, including integration with some
concrete corpus objects.

"""

import concurrent.futures as cf
import csv
import multiprocessing as mp
import pathlib
import shutil
import uuid
from datetime import date

import pytest

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
def fixture_example_index(tmp_path):
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
    total_docs = idx.n_docs()
    all_doc_ids = idx.all_docs()
    assert total_docs == target_docs == len(all_doc_ids)
    assert nnz == target_nnz

    idx.rebuild(doc_batch_size=10, index_positions=True)

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


def test_all_doc_counting(pool, example_index_corpora_path):

    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], corpus)


def test_field_feature_retrieval(example_index_corpora_path):
    """Confirm the field filtering works as expected."""

    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], corpus)

    test_combinations = (
        ({}, 2212),
        ({"min_docs_count": 10}, 229),
        ({"top_k": 100}, 100),
        ({"top_k": 100, "min_docs_count": 10}, 100),
    )

    for kwargs, expected_count in test_combinations:
        text_features = idx.field_features("text", **kwargs)
        assert all(
            docs_count >= kwargs.get("min_docs_count", 1)
            for feature, docs_count in text_features
        )
        assert len(text_features) == expected_count
        assert text_features[0][1] > text_features[-1][1]


def test_feature_querying(example_index_corpora_path, pool):
    """Test some simple applications of boolean querying and rendering of results."""
    corpus = hyperreal.corpus.PlainTextSqliteCorpus(example_index_corpora_path[0])
    idx = hyperreal.index.Index(example_index_corpora_path[1], corpus=corpus, pool=pool)

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
