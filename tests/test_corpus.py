"""
Basic test cases for inbuilt corpora.

Note that this is intentionally a small set of unit-tests: the tests here are
complementary to the more extensive tests provided through the notebooks that provide
more fleshed out examples of use cases (at the cost of being more expensive to run).

"""

import pathlib
import random

from hyperreal import corpus


def test_plaintext_corpus():

    moby_dick = corpus.TextfileParagraphsCorpus(
        pathlib.Path("tests", "data", "source", "moby_dick.txt")
    )

    assert len(moby_dick.paragraph_positions)
    assert sum(1 for _ in moby_dick.all_doc_keys())

    # Check everything is accessible and indexable:
    doc_count = sum(1 for _ in moby_dick.docs(moby_dick.all_doc_keys()))
    assert doc_count

    indexed = (
        moby_dick.doc_to_features(doc)
        for _, doc in moby_dick.docs(moby_dick.all_doc_keys())
    )
    indexed_count = sum(1 for _ in indexed)
    assert indexed_count
    assert doc_count == indexed_count

    # test random sampling of keys:
    n_docs = len(moby_dick)
    test_keys = [random.randint(0, n_docs) for _ in range(20)]

    assert sum(1 for _ in moby_dick.docs(test_keys)) == 20

    partial_docs = (
        moby_dick.doc_to_features(doc) for _, doc in moby_dick.docs(test_keys)
    )
    assert sum(1 for _ in partial_docs) == 20
