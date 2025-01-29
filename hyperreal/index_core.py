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
import itertools
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional

from pyroaring import AbstractBitMap, BitMap, BitMap64

from . import corpus, db_utilities, index_builder, value_handlers, index_plugin
from .feature_cluster import FeatureCluster


core_migration = index_plugin.Migration(
    None,
    "1",
    steps=[
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
            max_cardinality,
            value_handler_name,
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


class HyperrealIndex:

    def __init__(
        self,
        index_path,
        corpus: corpus.HyperrealCorpus,
        pool: Optional[cf.Executor],
        # TODO: document this -> can take either a class that is initialised with the
        # index, or a Callable that returns an IndexPlugin instance.
        plugins=(FeatureCluster,),
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

        try:
            self.db.execute("begin")
            self._ensure_migrated(CoreSchema(self))
            self._init_plugins()
            self.db.execute("commit")
        except Exception:
            self.db.execute("rollback")
            raise

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
    def field_handlers(self) -> dict[str, value_handlers.ValueHandler]:
        """
        Maps fields that have been indexed to appropriate value handlers

        This is the complement of corpus.schema, and allows mapping from field names
        back to handlers.

        """

        if not hasattr(self, "_field_handlers"):

            field_details = self.db.execute(
                """
                SELECT field, value_handler_name, max_cardinality, position_count 
                from field_summary
                """
            )

            _field_handlers = {}

            # TODO, process cardinalities to determine whether they were range encoded
            # or not.
            for field, name, cardinality, position_count in field_details:

                handler = self.corpus.name_handlers[name]

                range_encoded = (
                    cardinality,
                    position_count,
                    handler.stored_sorted,
                ) == (1, 0, True)

                _field_handlers[field] = (
                    handler,
                    range_encoded,
                )

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

    def rebuild(self, doc_batch_size: int = 1000, max_workers: Optional[int] = None):
        """(Re)build this index, indexing all documents on the corpus."""

        index_builder.build_index(
            self.index_path, self.corpus, self.pool, doc_batch_size, max_workers
        )

        # The schema might have changed, so invalidate if present.
        if hasattr(self, "_field_handlers"):
            del self._field_handlers

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

    def field_values(self, field: str) -> FieldValues:
        """
        Return the indexed values for field.

        This is the starting point for tabulating things like word counts, and
        displaying complex collections of values in different formats.

        """
        handler, range_encoded = self.field_handlers[field]

        rows = self.db.execute(
            """
            SELECT value, doc_count, position_count
            from inverted_index
            where field = ?
            order by value
            """,
            [field],
        )
        values, doc_counts, position_counts = list(zip(*rows))
        values = [handler.from_index(v) for v in values]

        if range_encoded:
            # Range encoded values need to be diffed, as they're a cumulative
            # sum
            doc_counts_diff = [doc_counts[0]]

            for d in doc_counts[1:]:
                doc_counts_diff.append(d - doc_counts_diff[-1])

            stats = {"doc_count": doc_counts_diff}

        else:
            stats = {"doc_count": doc_counts}

            if any(position_counts):
                stats["position_count"] = position_counts

        return FieldValues(
            field=field,
            values=values,
            statistics=stats,
            range_encoded=range_encoded,
        )

    def facet_count(self, field, queries: dict[str, AbstractBitMap], value_bins=None):
        """
        Count intersection of each of queries with the full set of values in field.

        Returns:

        - values of the facet
        - counts of each query intersection with that value facet
        - total size of the facet, for normalisation purposes

        """

        # Strategy: create a FieldValues object to drive the iteration for this
        # function.

        if value_bins is None:
            field_values = self.field_values(field)

        else:
            handler, range_encoded = self.field_handlers[field]
            ranges = [
                slice(left, right) for left, right in zip(value_bins, value_bins[1:])
            ]

        return

    def _match_literal_feature(self, field, index_value):

        matching = list(
            self.db.execute(
                "SELECT doc_ids from inverted_index where (field, value) = (?, ?)",
                (field, index_value),
            )
        )

        if matching:
            return matching[0][0]
        else:
            return BitMap()

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

    def _match_range_encoded_literal_feature(self, field, index_value):

        # Pick the literal value - if this isn't present then we can return immediately
        include_row = list(
            self.db.execute(
                "SELECT doc_ids from inverted_index where (field, value) = (?, ?)",
                (field, index_value),
            )
        )

        if not include_row:
            return BitMap()
        else:
            include_docs = include_row[0][0]

        # Now we retrieve the value that is just prior to the actual value.
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
            return include_docs - exclude_row[0][1]
        else:
            return include_docs

    def _match_slice_feature(self, field, index_value_start, index_value_end):

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
                SELECT roaring_union(doc_ids) 
                from inverted_index 
                {where_clause}
                """,
                args,
            )
        )

        if matching:
            return BitMap.deserialize(matching[0][0])
        else:
            return BitMap()

    def _match_slice_feature_pos(self, field, index_value_start, index_value_end):

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

    def _match_range_encoded_slice_feature(
        self, field, index_value_start, index_value_end
    ):

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

        return include_docs - exclude_docs

    def _iter_docs_for_features(self, field_values):
        """
        Turn a FieldValues Clause into an iterable of values and docs.

        This is responsible for turning a clause into an iterable of docs, one per
        matching value component.


        """
        # TODO: update nomenclature and make types clear
        # TODO: what is the public interface? Is it just match_any, match_all?
        # TODO: should I validate the types as well?

        if field_values.field not in self.field_handlers:
            raise ValueError(
                f"Field '{field_values.field}' does not exist on this index."
            )

        handler, range_encoded = self.field_handlers[field_values.field]

        # Map the lookup types based on how the field is encoded
        lookup_literal = self._match_literal_feature
        lookup_slice = self._match_slice_feature

        if range_encoded:
            lookup_literal = self._match_range_encoded_literal_feature
            lookup_slice = self._match_range_encoded_slice_feature

        for value in field_values.values:
            if isinstance(value, slice):
                if value.step is not None:
                    raise (ValueError("Step is not supported for a slice query."))

                start = None
                if value.start is not None:
                    start = handler.to_index(value.start)

                stop = None
                if value.stop is not None:
                    stop = handler.to_index(value.stop)
                yield value, lookup_slice(field_values.field, start, stop)
            else:
                yield value, lookup_literal(field_values.field, handler.to_index(value))

    def _iter_pos_for_features(self, field_values):

        if field_values.field not in self.field_handlers:
            raise ValueError(
                f"Field '{field_values.field}' does not exist on this index."
            )

        handler, _ = self.field_handlers[field_values.field]

        for value in field_values.values:
            if isinstance(value, slice):
                if value.step is not None:
                    raise (ValueError("Step is not supported for a slice query."))

                start = None
                if value.start is not None:
                    start = handler.to_index(value.start)

                stop = None
                if value.stop is not None:
                    stop = handler.to_index(value.stop)
                yield value, self._match_slice_feature_pos(
                    field_values.field, start, stop
                )
            else:
                yield value, self._match_literal_feature_pos(
                    field_values.field, handler.to_index(value)
                )

    def match_any(self, *field_value_clauses):
        """Evaluate an OR query across a set of FieldValues."""

        feature_docs = (
            (value, docs)
            for field_values in field_value_clauses
            for value, docs in self._iter_docs_for_features(field_values)
        )

        result_docs = BitMap()

        for _, docs in feature_docs:
            result_docs |= docs

        return result_docs

    def match_all(self, *field_value_clauses):
        """Evaluate an AND query"""
        feature_docs = (
            (value, docs)
            for field_values in field_value_clauses
            for value, docs in self._iter_docs_for_features(field_values)
        )

        _, result_docs = next(feature_docs)

        for _, docs in feature_docs:
            result_docs &= docs

        return result_docs

    def match_phrase(self, field_phrase):

        feature_positions = self._iter_pos_for_features(field_phrase)

        _, matches = next(feature_positions)
        current_shift = 1

        for _, next_positions in feature_positions:

            for already_matched in list(matches):

                test_match = already_matched + current_shift

                # TODO: index needs to keep track of the group_size when indexing...
                # This is currently hardcoded.
                group_shift, test_match = divmod(test_match, 8)

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
            return self.groups_to_docs(field_phrase.field, matching_groups)
        else:
            return BitMap()

    def groups_to_docs(self, field, positions):

        pos_docs, pos_doc_starts = list(
            self.db.execute(
                "SELECT group_doc_ids, doc_group_starts from field_summary where field = ?",
                [field],
            )
        )[0]

        docs = BitMap()

        for p in positions:
            docs.add(pos_docs[pos_doc_starts.rank(p) - 1])

        return docs


@dc.dataclass
class FieldValues:
    # Clause or Expression orr FeatureGroup or FieldValuesClause?
    # Empty value list matches all docs in field?
    # Empty field matches all docs?
    """
    The basic building block for querying and retrieving documents by their contents.

    """

    field: str
    values: list = dc.field(default_factory=list)
    statistics: collections.defaultdict[list] = dc.field(
        default_factory=lambda: collections.defaultdict(list)
    )
    range_encoded: bool = False

    def to_html(self, idx: HyperrealIndex):
        pass

    def sort_by(self, statistics_key: str, ascending=True):
        """
        Return a new FieldValues object with values sorted by the given statistic.

        """

        v

        return FieldValues(
            field=self.field,
        )


# Building queries? Build up clauses out of field values - these are the basic building
# blocks of complex queries - query nodes take
