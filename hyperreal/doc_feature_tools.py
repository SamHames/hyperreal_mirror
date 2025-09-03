"""
Toolkit for transforming and converting the indexable format of documents as mappings
of lists and features.

These tools are part of the core handling of concordances, snippets/passages and feature
matching display to contextualise search results.

"""

import collections


def match_features(doc_features, match_against):
    """Identify features from match_against that occur in doc_features."""
    literal_matches, range_matches = _split_all_literal_range_features(match_against)

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
    doc_features,
    match_against,
    window_size: int = 5,
    display_features=None,
):
    """
    Convert doc_features into a concordance of feature values at matching features.

    Only fields with positional information (ie, using lists of values) will be matched.

    This is a building block for counting cooccurrence/collocation of features, and also
    for the construction of concordances.

    """
    neighbourhoods = collections.defaultdict(list)

    display_features = display_features or {}

    for field, values in doc_features.items():

        if isinstance(values, list):
            # Use display values only if it's present for this field
            display_values = display_features.get(field, values)

            literal_matches, range_matches = _split_field_literal_range_values(
                match_against, field=field
            )

            # Match against the doc_features as that's the index form.
            if range_matches:
                match_locations = [
                    pos
                    for pos, val in enumerate(values)
                    if val in literal_matches or _match_range_value(val, range_matches)
                ]
            else:
                match_locations = [
                    pos for pos, val in enumerate(values) if val in literal_matches
                ]

            for match_loc in match_locations:

                pre = display_values[max(0, match_loc - window_size) : match_loc]
                post = display_values[match_loc + 1 : match_loc + window_size + 1]
                match_value = values[match_loc]
                display_match = display_values[match_loc]

                neighbourhoods[(field, match_value)].append(
                    (match_loc, pre, display_match, post)
                )

    return neighbourhoods


def to_passages(
    doc_features,
    passage_starts: dict[str, list[int]],
    passage_size: int,
    display_features=None,
):
    """
    Extract passages of values at given locations from the positional fields.

    Used to create snippets and subsets of document information for display.

    """
    passages = collections.defaultdict(list)
    display_features = display_features or {}

    for field, values in doc_features.items():

        if isinstance(values, list):
            # Use display values only if it's present for this field
            display_values = display_features.get(field, values)

            max_position = len(values)

            for start in passage_starts.get(field, []):
                end = min(max_position, start + passage_size)
                passages[field].append((start, start + passage_size, values[start:end]))

    return passages


def _match_range_value(value, match_ranges):

    # Note we're looking for any positive match! Only one range needs to match to be
    # True, everything needs to be False to be false.
    for start, end in match_ranges:

        if start is None:
            if value < end:
                return True
        elif end is None:
            if value >= start:
                return True
        else:
            if start <= value < end:
                return True

    return False


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
