"""
Basic test cases for tools that transform the indexable doc format in various ways.

"""

from hyperreal import doc_feature_tools as dft


test_doc_with_positional = {
    # positional field
    "text": "the cat sat on the mat with theodore".split(),
    "date": "2020-01-01",
}


test_doc_with_no_positional = {
    "date": "2020-01-01",
    "integer": 1,
    "float": 2.0,
    "set": set(range(3)),
}


def test_converted():

    assert len(dft.to_feature_set(test_doc_with_positional)) == 8

    assert len(dft.to_feature_set(test_doc_with_no_positional)) == 6


def test_neighbourhoods():

    for match, expected in (("the", 2), ("cat", 1), ("asld;kfjsdaljk", 0)):

        to_match = [("text", match)]
        neighbourhoods = dft.to_matching_neighbourhood(
            test_doc_with_positional, to_match, window_size=1
        )

        assert len(neighbourhoods["text"]) == expected


def test_neighbourhoods_range():

    for match, expected in (
        (("the", "the\u10FFFF"), 3),
        # Exclusive upper bound, won't match cat.
        ((None, "cat"), 0),
        # Will match everything, inclusive lower bound
        (("cat", None), 8),
    ):

        to_match = [("text", *match)]
        neighbourhoods = dft.to_matching_neighbourhood(
            test_doc_with_positional, to_match, window_size=1
        )

        assert len(neighbourhoods["text"]) == expected


def test_simple_match():

    matching = dft.match_features(
        test_doc_with_positional, [("text", "the"), ("date", "2020-01-01")]
    )
    assert len(matching) == 2

    matching = dft.match_features(
        test_doc_with_no_positional, [("text", "the"), ("date", "2020-01-01")]
    )
    assert len(matching) == 1


def test_range_match():

    matching = dft.match_features(
        test_doc_with_positional,
        [
            # Range query on the text the* - should match the and theodore
            ("text", "the", "the\u10FFFF"),
            # Should not match as upper bound is exclusive
            ("date", "2019-01-01", "2020-01-01"),
        ],
    )
    assert len(matching) == 2

    matching = dft.match_features(
        test_doc_with_no_positional,
        [
            ("integer", 0, 10),
            ("float", 1.0, 1.5),
            ("integer", None, 0),
            ("set", 1, None),
        ],
    )
    assert len(matching) == 3


def test_passages():

    # Date will be ignored because it's not positional.
    passage_starts = {"text": [1], "date": [1]}

    passages = dft.to_passages(test_doc_with_positional, passage_starts, passage_size=1)
    assert len(passages["text"]) == 1
    assert len(passages["date"]) == 0
