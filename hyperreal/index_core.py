"""

Main interface for working with an index: the HyperrealIndex class itself.


Examples, setup, usage, general ideas.

Interactions between the schema and this?

How does a query work?

Extending the query interface.

Documentation page: querying.


Example for doc tests: basic functionality of all the methods.

"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import dataclasses as dc
from functools import cached_property
import itertools
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, Hashable

from pyroaring import AbstractBitMap, BitMap, BitMap64
from tinyhtml import h, raw, frag

from . import corpus, db_utilities, index_builder, value_handlers, index_plugin
from .feature_cluster import FeatureClustering


core_migration = index_plugin.Migration(
    None,
    "1",
    steps=[
        """
        CREATE table index_setting (
            /* Hold settings information (typically generated at build time) */
            key text primary key,
            value
        );
        """,
        """
        CREATE table doc_key (
            doc_id integer primary key,
            doc_key unique
        )
        """,
        """
        CREATE table field_summary(
            field text primary key,
            -- TODO: cardinality is potentially confusing, because we can talk about it
            -- here as 1:x cardinality, or separately as the cardinality/unique values
            -- defined on a field. Is there a more precise naming for each?
            -- TODO: we also want the number of unique values on a field for display
            -- purposes.
            value_handler_name,
            max_doc_cardinality,
            unique_value_count,
            min_value,
            max_value,
            stored_sorted,
            doc_count,
            doc_ids roaring_bitmap,
            position_count,
            group_doc_ids roaring_bitmap,
            doc_group_starts roaring_bitmap
        )
        """,
        """
        CREATE table inverted_index(
            field text references field_summary,
            value not null,
            doc_count,
            doc_ids roaring_bitmap,
            position_count,
            primary key (field, value)
        )
        """,
        """
        CREATE table position_index(
            field text references field_summary,
            value not null,
            mod_position integer not null,
            position_count,
            group_ids roaring_bitmap,
            primary key (field, value, mod_position)
            foreign key (field, value) references inverted_index
        )
        """,
    ],
    description="Initialise the core tables for holding the index of the corpus.",
)


class CoreSchema(index_plugin.IndexPlugin):
    plugin_name = "hyperreal_core"
    current_version = "1"
    migrations = [core_migration]


LiteralFeature = tuple[str, Hashable]
RangeFeature = tuple[str, Optional[Hashable], Optional[Hashable]]
Feature = LiteralFeature | RangeFeature

FeatureStatistics = dict[Feature, dict[str, int | float]]


class HyperrealIndex:

    def __init__(
        self,
        index_path,
        corpus: corpus.HyperrealCorpus,
        pool: Optional[cf.Executor],
        # TODO: document this -> can take either a class that is initialised with the
        # index, or a Callable that returns an IndexPlugin instance.
        plugins=(FeatureClustering,),
    ):
        """Plugins are initialised once at startup."""

        self.index_path = index_path
        self.corpus = corpus
        self.pool = pool
        self.db = db_utilities.connect_sqlite(index_path)
        self._provided_plugins = plugins or []
        self.p = SimpleNamespace()

        # TODO: validate all the details of the plugins are consistent - unique
        # plugin_names etc.

        # This is the minimum necessary to bootstrap everything else - we manage the
        # core db schema state just like any other plugins state.
        self.db.execute(
            """
            CREATE table if not exists plugin_version (
                plugin_name text primary key,
                version
            )
            """
        )
        self.db.execute("pragma journal_mode=WAL")
        self.db.execute("pragma foreign_keys=ON")

        try:
            self.db.execute("begin")
            self._ensure_migrated(CoreSchema(self))
            self._init_plugins()
            self.db.execute("commit")
        except Exception:
            self.db.execute("rollback")
            raise

        self._field_handlers = None
        self._passage_group_size = None

    def __enter__(self):
        self.db.execute("begin")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on exception, otherwise commit.
            self.db.execute("rollback")
        else:
            self.db.execute("commit")

    def close(self):
        self.db.close()

    def _init_plugins(self):
        """Initialise plugins, ensuring all migrations have been run."""

        for PluginClass in self._provided_plugins:

            # Initialise the plugin class with this index.
            plugin = PluginClass(self)

            # Check version against the index and run migrations if necessary.
            self._ensure_migrated(plugin)

            # Finally make the plugin available against the appropriate namespace in the
            # index.
            setattr(self.p, plugin.plugin_name, plugin)

    def _ensure_migrated(self, plugin: index_plugin.IndexPlugin):

        index_version = self.get_plugin_version(plugin.plugin_name)

        while index_version != plugin.current_version:

            migration = plugin.version_migration_map[index_version]
            index_version = migration.to_version
            # TODO: log migrations being applied
            # TODO: logging framework in general...
            self._apply_migration(migration)
            # TODO: error message when index version not in possible migrations

        self._set_plugin_version(plugin.plugin_name, index_version)

    def _apply_migration(self, migration: index_plugin.Migration):

        for i, step in enumerate(migration.steps):
            # TODO: log and raise meaningful error
            if isinstance(step, str):
                self.db.execute(step)
            elif callable(step):
                step(self)

    def get_plugin_version(self, plugin_name: str):
        version = list(
            self.db.execute(
                "select version from plugin_version where plugin_name = ?",
                [plugin_name],
            )
        )

        if version:
            return version[0][0]
        else:
            return None

    def _set_plugin_version(self, plugin_name: str, version: str):
        """
        Should only be called by init plugins.
        """
        self.db.execute(
            "replace into plugin_version values (?, ?)", [plugin_name, version]
        )

    @property
    def passage_group_size(self) -> int:
        """The defined passage group size from the indexing run."""

        if self._passage_group_size is None:
            result = list(
                self.db.execute(
                    "SELECT value from index_setting where key = ?",
                    ["passage_group_size"],
                )
            )

            if result:
                self._passage_group_size = result[0][0]

        return self._passage_group_size

    @property
    def field_handlers(self) -> dict[str, value_handlers.ValueHandler]:
        """
        Maps fields that have been indexed to appropriate value handlers

        This is the complement of corpus.schema, and allows mapping from field names
        back to handlers.

        """

        if self._field_handlers is None:

            field_details = self.db.execute(
                """
                SELECT field, value_handler_name, max_doc_cardinality, position_count 
                from field_summary
                """
            )

            _field_handlers = {}

            for field, name, cardinality, position_count in field_details:

                handler = self.corpus.name_handlers[name]

                range_encoded = (
                    cardinality,
                    position_count,
                    handler.stored_sorted,
                ) == (1, 0, True)

                _field_handlers[field] = (handler, range_encoded, cardinality)

            self._field_handlers = _field_handlers

        return self._field_handlers

    def __getstate__(self):
        return self.index_path, self.corpus

    def __setstate__(self, args):
        """
        Used to handle pickling the index object for use of an index in a process pool.

        Note that the process pool and plugins are not available on the pickled object?

        """
        # TODO: think about whether plugins should be available on a pickled index.
        self.__init__(args[0], args[1], None)

    def rebuild(
        self,
        doc_batch_size: int = 1000,
        max_workers: Optional[int] = None,
        passage_group_size=16,
    ):
        """(Re)build this index, indexing all documents on the corpus."""

        index_builder.build_index(
            self.index_path,
            self.corpus,
            self.pool,
            doc_batch_size=doc_batch_size,
            max_workers=max_workers,
            passage_group_size=passage_group_size,
        )

        # The schema might have changed, so invalidate if present.
        self._field_handlers = None
        self._passage_group_size = passage_group_size
        del self.indexed_field_summary

    @cached_property
    def indexed_field_summary(self):

        header = (
            "Field",
            "Value Type",
            "cardinality?",
            "Unique Values",
            "Mininum Value",
            "Maximum Value",
            "Number of Documents",
            "Number of Positions",
            "Sortable",
            "Range Encoded",
        )

        rows = list(
            list(row)
            for row in self.db.execute(
                """
                SELECT 
                    field, 
                    value_handler_name, 
                    max_doc_cardinality, 
                    unique_value_count,
                    min_value,
                    max_value,
                    doc_count,
                    position_count,
                    stored_sorted,
                    stored_sorted = 1 and max_doc_cardinality = 1 as range_encoded 
                from field_summary
                order by field
                """
            )
        )

        for row in rows:
            field = row[0]
            handler = self.field_handlers[field][0]

            min_value, max_value = row[4:6]

            row[4:6] = handler.from_index(min_value), handler.from_index(max_value)
            row[-2] = bool(row[-2])
            row[-1] = bool(row[-1])

        return [header] + [tuple(row) for row in rows]

    @property
    def max_doc_id(self) -> Optional[int]:
        """
        The maximum doc_id in the collection.

        If no docs are present in the collection, returns None.

        """
        if self.total_doc_count == 0:
            return None
        else:
            return self.total_doc_count - 1

    @property
    def total_doc_count(self) -> int:
        """Return the number of indexed documents."""
        return list(
            self.db.execute("select coalesce(max(doc_id) + 1, 0) from doc_key")
        )[0][0]

    def all_doc_ids(self) -> BitMap:
        """Returns all doc_ids in the collection."""
        return BitMap(range(0, self.total_doc_count))

    def _iter_corpus_docs(self, doc_ids: Iterable[int], corpus_method: Callable):
        """
        Helper function for iterating through transformed documents from a corpus.

        This handles generating an iterator of doc_keys, and keeping that in sync
        with the doc_ids.

        """

        # This is implemented in a fiddly way, as the corpus should only take an
        # iterator of keys, but we want to keep the doc_id -> doc_key links intact
        # without loading everything into memory.
        doc_keys = self.doc_ids_to_keys(doc_ids)

        # Split the generator into two, so we can iterate and keep everything aligned.
        for_corpus, for_doc_id = itertools.tee(doc_keys, 2)
        corpus_docs = corpus_method((doc_key for _, doc_key in for_corpus))

        # Walk through the corpus docs, potentially allowing the corpus to swallow
        # keys for any reason (such as for a missing doc).
        for doc_key1, doc in corpus_docs:

            # Advance through the other iterator until we get the same doc_key. It is
            # assumed that this will almost always just be the very next one/ that
            # missing keys are rare.
            for doc_id, doc_key2 in for_doc_id:
                if doc_key1 == doc_key2:
                    break
            # TODO - this could be less fiddly if there are constraints on what a corpus
            # can do with keys. This interface allows a corpus to do something like
            # handle a missing key by continuing on to the next doc.
            yield doc_id, doc_key1, doc

    def docs(self, doc_ids):
        """
        Iterate through documents on the corpus matching doc_ids.

        """

        return self._iter_corpus_docs(doc_ids, self.corpus.docs)

    def indexable_docs(self, doc_ids):
        """
        Iterate through indexable form of documents on the corpus matching doc_ids.

        """

        return self._iter_corpus_docs(doc_ids, self.corpus.indexable_docs)

    def html_docs(self, doc_ids):
        """
        Iterate through HTML form of documents on the corpus matching doc_ids.

        """

        return self._iter_corpus_docs(doc_ids, self.corpus.html_docs)

    def _lookup_doc_key(self, doc_id: int):
        """
        Return the doc_key for the given doc_id.

        It's usually preferable to use the bulk forms doc_ids_to_keys or the docs and
        other methods directly. This method does not handle the case where doc_id is
        not present on the index.

        """

        return list(
            self.db.execute("SELECT doc_key from doc_key where doc_id = ?", [doc_id])
        )[0][0]

    def doc_ids_to_keys(self, doc_ids: Iterable[int]) -> Iterable[tuple[int, Any]]:
        """Generate document keys on the corpus from the given doc_ids."""

        self.db.execute("savepoint doc_ids_to_keys")

        # Validate doc_ids are valid. Since doc_ids are all contiguous this is
        # straightforward.
        total_docs = self.total_doc_count

        for doc_id in doc_ids:
            # total_docs is exclusive bound, as doc_id's start at 0, so max_doc_id
            # is one less than total_docs
            if not (0 <= doc_id < total_docs):
                raise ValueError(
                    f"Invalid {doc_id=} - valid doc_id's for this index are between 0 "
                    f"and {max_doc_id}"
                )

            yield doc_id, self._lookup_doc_key(doc_id)

        self.db.execute("release doc_ids_to_keys")

    def field_features(self, field: str, min_docs: int = 1) -> FeatureStatistics:
        """
        Return the indexed features for field.

        The keys of the returned dictionary are the features, the values are
        dictionaries describing the statistics of the features.

        TODO: document link to tabulation.

        """

        handler, range_encoded, _ = self.field_handlers[field]

        features = {}

        if range_encoded:

            rows = self.db.execute(
                """
                SELECT value, doc_count
                from inverted_index
                where field = ?
                order by value
                """,
                [field],
            )

            previous_doc_count = 0

            for value, cumulative_doc_count in rows:
                doc_count = cumulative_doc_count - previous_doc_count
                if doc_count >= min_docs:
                    feature = (field, handler.from_index(value))
                    stats = {"doc_count": doc_count}
                    features[feature] = stats

                previous_doc_count = cumulative_doc_count

        else:
            rows = self.db.execute(
                """
                SELECT value, doc_count, position_count
                from inverted_index
                where field = ?
                    and doc_count >= ?
                order by value
                """,
                [field, min_docs],
            )
            for value, doc_count, position_count in rows:
                feature = (field, handler.from_index(value))
                stats = {"doc_count": doc_count, "position_count": position_count}

                features[feature] = stats

        return features

    def facet_features(
        self, features: Iterable[Feature], query: AbstractBitMap
    ) -> FeatureStatistics:
        """
        Intersect the docs matching each feature with the provided query.

        Provides a number of statistics about the relationship between the given query
        and fields.

        Returns:

        """

        results = dict()

        q_len = len(query)
        total_docs = self.total_doc_count

        for feature in features:

            stats = {}

            docs, doc_count, position_count = self[feature]

            stats["doc_count"] = doc_count

            inter = query.intersection_cardinality(docs)
            stats["hits"] = inter

            # Compute some other derived statistics
            stats["jaccard_similarity"] = inter / (doc_count + q_len - inter)
            stats["feature_proportion"] = inter / doc_count
            stats["query_proportion"] = inter / q_len

            results[feature] = stats

        return results

    def __getitem__(self, feature):
        """
        Retrieve the set of documents containing the given feature.

        """
        field = feature[0]
        value_spec = feature[1:]

        try:
            handler, range_encoded, _ = self.field_handlers[field]
        except KeyError:
            raise KeyError(f"Field '{field}' does not exist on this index.")

        # Map the lookup types based on how the field is encoded
        lookup_literal = self._match_literal_feature
        lookup_range = self._match_range_feature

        if range_encoded:
            lookup_literal = self._match_range_encoded_literal_feature
            lookup_range = self._match_range_encoded_range_feature

        if len(value_spec) == 1:  # Literal feature
            index_value = handler.to_index(value_spec[0])
            return lookup_literal(field, index_value)

        elif len(value_spec) == 2:  # Range feature
            value_start, value_end = value_spec

            if value_start is not None:
                index_start = handler.to_index(value_start)
            if value_end is not None:
                index_end = handler.to_index(value_end)

            return lookup_range(field, index_start, index_end)
        else:
            raise ValueError(
                "A feature can be at most three elements long "
                f"(field, range_start, range_end) - got {feature}"
            )

    def _match_literal_feature(self, field, index_value) -> tuple[BitMap, int, int]:

        matching = list(
            self.db.execute(
                """
                SELECT 
                    doc_ids, doc_count, position_count 
                from inverted_index 
                where (field, value) = (?, ?)
                """,
                (field, index_value),
            )
        )

        if matching:
            return matching[0]
        else:
            return BitMap(), 0, 0

    def _match_literal_feature_pos(self, field, index_value):

        matching_rows = self.db.execute(
            """
            SELECT 
                mod_position, 
                group_ids 
            from position_index 
            where (field, value) = (?, ?)
            """,
            (field, index_value),
        )

        return {mod_position: group for mod_position, group in matching_rows}

    def _match_range_encoded_literal_feature(
        self, field, index_value
    ) -> tuple[BitMap, int, int]:

        # Pick the literal value - if this isn't present then we can return immediately
        include_row = list(
            self.db.execute(
                "SELECT doc_ids from inverted_index where (field, value) = (?, ?)",
                (field, index_value),
            )
        )

        if not include_row:
            return BitMap(), 0, 0
        else:
            include_docs = include_row[0][0]

        # Now we retrieve the value that is just prior to the actual value.
        # This may not exist if we retrieved the smallest value for this field.
        exclude_row = list(
            self.db.execute(
                """
                SELECT max(value), doc_ids 
                from inverted_index 
                where field = ?
                    and value < ?
                """,
                (field, index_value),
            )
        )

        if exclude_row:
            include_docs -= exclude_row[0][1]

        # Note that the position count is always zero for a range-encoded field.
        return include_docs, len(include_docs), 0

    def _match_range_feature(
        self, field, index_value_start, index_value_end
    ) -> tuple[BitMap, int, int]:

        if index_value_start is None and index_value_end is None:
            raise ValueError(
                "One of index_value_start and index_value_end needs to be not None."
            )
        elif index_value_start is None:
            where_clause = "where field = ? and value < ?"
            args = [field, index_value_end]
        elif index_value_end is None:
            where_clause = "where field = ? and value >= ?"
            args = [field, index_value_start]
        else:
            where_clause = "where field = ? and value >= ? and value < ?"
            args = [field, index_value_start, index_value_end]

        matching = list(
            self.db.execute(
                f"""
                SELECT roaring_union(doc_ids), sum(position_count)
                from inverted_index 
                {where_clause}
                """,
                args,
            )
        )

        if matching:
            matching_docs = BitMap.deserialize(matching[0][0])
            return matching_docs, len(matching_docs), matching[0][1]
        else:
            return BitMap(), 0, 0

    def _match_range_feature_pos(self, field, index_value_start, index_value_end):

        if index_value_start is None and index_value_end is None:
            raise ValueError(
                "One of index_value_start and index_value_end needs to be not None."
            )
        elif index_value_start is None:
            where_clause = "where field = ? and value < ?"
            args = [field, index_value_end]
        elif index_value_end is None:
            where_clause = "where field = ? and value >= ?"
            args = [field, index_value_start]
        else:
            where_clause = "where field = ? and value >= ? and value < ?"
            args = [field, index_value_start, index_value_end]

        matching = self.db.execute(
            f"""
            SELECT mod_position, roaring_union(groups) 
            from position_index 
            {where_clause}
            group by mod_position
            """,
            args,
        )

        return {
            mod_position: BitMap.deserialize(groups)
            for mod_position, groups in matching
        }

    def _match_range_encoded_range_feature(
        self, field, index_value_start, index_value_end
    ) -> tuple[BitMap, int, int]:

        if index_value_start is None and index_value_end is None:
            raise ValueError(
                "One of index_value_start and index_value_end needs to be not None."
            )

        if index_value_start is None:
            # Easy case - no lower bound documents to exclude at the end
            exclude_row = BitMap()

        else:
            # We actually have to check if there are lower bound docs so we can exclude
            # them. It might happen that that value_start < smallest value in the index,
            # in which case there's nothing to exclude.
            exclude_row = list(
                self.db.execute(
                    """
                    SELECT max(value), doc_ids 
                    from inverted_index 
                    where field = ? and value < ?
                    """,
                    [field, index_value_start],
                )
            )

            if exclude_row:
                exclude_docs = exclude_row[0][1]

        if index_value_end is None:
            # Use the last row as the indicator of all docs matching this field.
            include_docs = list(
                self.db.execute(
                    """
                    SELECT max(value), doc_ids 
                    from inverted_index 
                    where field = ?
                    """,
                    [field],
                )
            )[0][1]

        else:
            # We actually have to check an upper bound value against the index values.
            # Because this is an exclusive bound, we do the same test as the lower
            # bound.
            include_row = list(
                self.db.execute(
                    """
                    SELECT max(value), doc_ids 
                    from inverted_index 
                    where field = ? and value < ?
                    """,
                    [field, index_value_end],
                )
            )

            if include_row:
                include_docs = include_row[0][1]

        matching_docs = include_docs - exclude_docs
        return matching_docs, len(matching_docs), 0

    def _iter_pos_for_features(self, field, features):

        # Assumes that features has already been validated for a single field
        if field not in self.field_handlers:
            raise ValueError(f"Field '{field}' does not exist on this index.")

        handler, _, _ = self.field_handlers[field]

        for field, value in features:
            if isinstance(value, slice):
                if value.step is not None:
                    raise (ValueError("Step is not supported for a slice query."))

                start = None
                if value.start is not None:
                    start = handler.to_index(value.start)

                stop = None
                if value.stop is not None:
                    stop = handler.to_index(value.stop)

                yield value, self._match_slice_feature_pos(field, start, stop)

            else:
                yield value, self._match_literal_feature_pos(
                    field, handler.to_index(value)
                )

    def match_phrase(self, *features):
        """
        Identify documents with the given features occuring in sequence (as a phrase).

        The set of features must all be on the same field. The features will be
        interpreted as the individual features of a phrase, in the iteration order of
        a container.

        """

        if isinstance(features, set):
            raise TypeError()

        fields = {f[0] for f in features}
        if len(fields) > 1:
            raise ValueError("All features must be on the same field for a phrase.")

        field = fields.pop()

        feature_positions = self._iter_pos_for_features(field, features)

        _, matches = next(feature_positions)
        current_shift = 1

        for _, next_positions in feature_positions:

            for already_matched in list(matches):

                test_match = already_matched + current_shift

                # TODO: index needs to keep track of the group_size when indexing...
                # This is currently hardcoded.
                group_shift, test_match = divmod(test_match, self.passage_group_size)

                if group_shift:
                    next_group_match = next_positions.get(test_match, BitMap())
                    matches[already_matched] &= next_group_match.shift(-group_shift)
                else:
                    matches[already_matched] &= next_positions.get(test_match, BitMap())

            # Remove any empty matches so they don't need to be checked.
            matches = {
                mod_position: group for mod_position, group in matches.items() if group
            }

            current_shift += 1

        if matches:
            matching_groups = BitMap.union(*matches.values())
            return self.groups_to_docs(field, matching_groups)
        else:
            return BitMap()

    def groups_to_docs(self, field: str, group_positions) -> BitMap:
        """Convert a set of positional matches to doc_ids."""

        group_docs, doc_group_starts = list(
            self.db.execute(
                """
                SELECT 
                    group_doc_ids, 
                    doc_group_starts 
                from field_summary 
                where field = ?
                """,
                [field],
            )
        )[0]

        docs = BitMap()

        for g in group_positions:
            docs.add(group_docs[doc_group_starts.rank(g) - 1])

        return docs


# Corpus interface for specifying facets to compute:
# 1. A callable that returns keys in order that they should be rendered.
# 2. A concrete set of features to render, or a callable to retrieve those features?
#
# Example Use cases?
# 1. Show me the timeline trend by month of this particular things
# 2. Show me the top 20 most common speakers who used this word
# 3. Show me the 20 most similar time periods to this query


class KeyedStatsSelector:
    def __init__(
        self,
        reverse=True,
        first_k=None,
    ):
        """
        Specifies how to select from and order a KeyedStats selector.

        """

    def __call__(self, feature_stats):
        return


class FeatureStatsSorterFilterer:
    def __init__(
        self,
        idx,
        order_by_field_value=None,
        order_by_stat=None,
        reverse=True,
        top_k=None,
        display_stats=None,
        drop_stat_values=None,
    ):
        """
        Specifies how to sort and order the keys in a feature stats object.

        Can also produce some

        """
        self.idx = idx

        if top_k is not None and order_by_stat is None:
            raise ValueError("order_by must be specified for top_k")

        self.order_by_stat = order_by_stat
        self.reverse = reverse
        self.top_k = top_k
        self.display_stats = display_stats

        self.field_handlers = {
            field: handler for field, (handler, _, _) in idx.field_handlers.items()
        }

        # Should this be a dictionary to sets, one per statistic?
        # self.drop_stat_values = drop_stat_values or set()

        # Probably also need something to apply a custom class? Ie, to differentiate
        # a facet from a cluster, from a list of clusters?

    def key_render_order(self, stats):
        """
        Return the list of keys to render and their order.

        This applies:

            - sorting by the selected statistic
            - selection of the top_k keys
            - dropping any keys with a statistic value in drop_stat_values
            - truncation of values?

        """
        # Drop values *before* sorting
        # if self.drop_stat_values:
        #     keep_keys = [key for key ]

        if self.order_by_stat is not None:
            key_order = sorted(
                stats.keys(),
                key=lambda k: stats[k][self.order_by_stat],
                reverse=self.reverse,
            )

            if self.top_k is not None:
                key_order = key_order[: self.top_k]
        else:
            key_order = stats.keys()

        return key_order

    def to_rows(self, stats):

        key_order = self.key_render_order(stats)

        header = ["field", "value", "value_end", *self.display_stats]
        rows = []

        for key in key_order:
            field = key[0]
            values = key[1:]

            if len(values) == 1:
                value = [*values, None, None]
            else:
                value = [None, *values]

            key_stats = stats[key]
            stats_row = [key_stats[s] for s in self.display_stats]

            row = [field, *value, *stats_row]

            rows.append(row)

        return rows

    def to_html_dl(self, stats):

        key_order = self.key_render_order(stats)

        last_field = None

        elements = []

        for key in key_order:
            field = key[0]
            values = key[1:]

            if field != last_field:
                elements.append(h("dt")(field))

            if len(values) == 1:
                elements.append(h("dd")(self.field_handlers[field].to_html(values[0])))
            elif len(values) == 2:
                elements.append(
                    h("dd")(
                        self.field_handlers[field].to_html(values[0]),
                        ":",
                        self.field_handlers[field].to_html(values[1]),
                    )
                )

            last_field = field

        return h("dd")(elements)

    def to_html_table(self, stats):
        key_order = self.key_render_order(stats)

        for feature in key_order:
            pass

    def to_str(self, stats):
        pass
