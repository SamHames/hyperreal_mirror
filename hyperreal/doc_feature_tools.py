"""
Toolkit for transforming and converting the indexable format of documents as mappings
of lists and features.

"""

import collections

from .corpus import IndexableDoc


def match_features(doc_features, match_against):
    """Identify features from match_against that occur in doc_features."""
    literal_matches, range_matches = _split_all_literal_range_features(match_against)

    matching = set()

    all_features = to_feature_set(doc_features)

    matching = all_features & literal_matches

    if range_matches:
        matching |= {
            feature
            for feature in all_features
            if _match_range_feature(feature, range_matches)
        }

    return matching


def to_feature_set(doc_features):
    """Convert doc_features to the unique set of features across all fields."""
    feature_set = set()

    for field, values in doc_features.items():
        if isinstance(values, (list, set)):
            feature_set |= {(field, v) for v in values}
        else:
            feature_set.add((field, values))

    return feature_set


def to_matching_neighbourhood(
    doc_features: IndexableDoc,
    match_against,
    window_size: int = 5,
):
    """
    Convert doc_features into a concordance of feature values at matching features.

    Only fields with positional information (ie, using lists of values) will be matched.

    This is a building block for counting cooccurrence/collocation of features, and also
    for the construction of concordances.

    """
    neighbourhoods = collections.defaultdict(list)

    for field, values in doc_features.items():

        if isinstance(values, list):

            literal_matches, range_matches = _split_field_literal_range_values(
                match_against, field=field
            )

            if range_matches:
                matches = [
                    pos
                    for pos, val in enumerate(values)
                    if val in literal_matches or _match_range_value(val, range_matches)
                ]
            else:
                matches = [
                    pos for pos, val in enumerate(values) if val in literal_matches
                ]

            for match in matches:

                pre = values[max(0, match - window_size) : match]
                post = values[match + 1 : match + window_size + 1]
                match_value = values[match]
                neighbourhoods[field].append((match, pre, match_value, post))

    return neighbourhoods


def to_passages(
    doc_features: IndexableDoc,
    passage_starts: dict[str, list[int]],
    passage_size: int,
):
    """
    Extract passages of values at given locations from the positional fields.

    Used to create snippets and subsets of document information for display.

    """
    passages = collections.defaultdict(list)

    for field, values in doc_features.items():

        if isinstance(values, list):

            max_position = len(values)
            for start in passage_starts[field]:
                end = min(max_position, start + passage_size)
                passages[field].append((start, start + passage_size, values[start:end]))

    return passages


def _match_range_value(value, match_ranges):

    for start, end in match_ranges:

        if start is None:
            return value < end
        elif end is None:
            return value >= start
        else:
            return start <= value < end


def _match_range_feature(feature, match_ranges):

    field, value = feature

    return _match_range_value(value, match_ranges[field])


def _split_field_literal_range_values(to_match, field):
    """
    Split to_match into literal and range features.

    """
    literal_value_matches = {
        feature[1] for feature in to_match if len(feature) == 2 and feature[0] == field
    }

    range_value_matches = {
        feature[1:] for feature in to_match if len(feature) == 3 and feature[0] == field
    }

    return literal_value_matches, range_value_matches


def _split_all_literal_range_features(to_match):

    literal_matches = set()
    range_matches = collections.defaultdict(set)

    for feature in to_match:
        if len(feature) == 2:
            literal_matches.add(feature)
        elif len(feature) == 3:
            range_matches[feature[0]].add(feature[1:])
        else:
            raise ValueError(
                f"Invalid {feature=}, a feature can only have two or three elements."
            )

    return literal_matches, range_matches
