"""

Main interface for working with an index: the HyperrealIndex class itself.


Examples, setup, usage, general ideas.

Interactions between the schema and this?

How does a query work?

Extending the query interface.

Documentation page: querying.


Example for doc tests: basic functionality of all the methods.

"""

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
        CREATE table if not exists field_summary(
            field text primary key,
            max_cardinality,
            value_handler_name,
            stored_sorted,
            doc_count,
            doc_ids roaring_bitmap,
            position_count,
            position_doc_ids roaring_bitmap,
            position_doc_starts roaring_bitmap64
        )
        """,
        """
        CREATE table if not exists inverted_index(
            field text references field_summary,
            value not null,
            doc_count,
            doc_ids roaring_bitmap,
            position_count,
            positions roaring_bitmap64,
            primary key (field, value)
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
        corpus: corpus.Corpus,
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

    def facet_count(self, field, queries, value_bins=None):
        """
        Return the counts for each query in queries against all values in fields, or all
        ranges in value_bins.

        """
        if value_bins is None:
            pass

        else:
            ranges = [
                slice(left, right) for left, right in zip(value_bins, value_bins[1:])
            ]
        return

    def __getitem__(self, field: str, value) -> BitMap:
        """
        Retrieve documents matching a literal field value, or a range of values.

        This is the most basic retrieval operation for documents based on their
        constituent features as fields and values.

        """

        # Validation 1: does this field exist for indexed documents
        if field not in self.field_handlers:
            raise ValueError(f"{field=} does not occur in this index.")

        handler, range_encoded = self.field_handlers[field]

        is_slice = isinstance(value, slice)

        if is_slice and not handler.stored_sorted:
            raise TypeError(
                f"{handler=} does not indicate that values are stored "
                "lexicographically sorted, range queries are not supported."
            )

        # Four cases (slice, literal) x (range_encoded, literal)
        if range_encoded:
            if is_slice:
                pass
            else:
                pass
        else:
            if is_slice:
                pass
            else:
                pass

        """
        Errors and validation:

        - field needs to be in field handlers, otherwise error
        - missing value is fine as long as the field exists: just return empty docs.
        - slices for ranges, but only on sortable fields. Step parameter must be None.
        
        """

    def _literal_feature_lookup(self, field, index_value):

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

    def _literal_range_encoded_feature_lookup(self, field, index_value):

        # Pick the literal value - if this isn't present then we can return immediately
        match_init = list(
            self.db.execute(
                "SELECT doc_ids from inverted_index where (field, value) = (?, ?)",
                (field, index_value),
            )
        )

        if not match_init:
            return BitMap()
        else:
            bm = match_init[0][0]

        # Now we retrieve the value that is just prior to the actual value.
        match_prev = list(
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

        if match_prev:
            return bm - match_prev[0][0]
        else:
            return bm

    def _slice_feature_lookup(self, field, index_value_start, index_value_end):

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

    def _slice_range_encoded_feature_lookup(
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

            print(exclude_row)

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

            print(include_row)

        return include_docs - exclude_docs

    def _iter_field_values(self, *values):
        """Iterate through docs matching each expression in field values."""

        for expression in values:
            # validate the field
            field = expression.field

    def _iter_docs_for_features(self, features):
        """"""
        # Feature specification: tuple (field, values)
        # dict (field: values)
        # single field?
        # iterator of any of the above.
        # Iterate through all the individual docs matching the given features?
        # TODO: some kind of cache for validating features and handling? Or precompute
        # what kind of features are supportable given something?

    def or_query(self, single_field_clause):
        """Evaluate an OR query across a single field of values."""
        pass

    def and_query(self, single_field_clause):
        """Evaluate an AND query"""


@dc.dataclass
class FieldValues:
    # Clause or Expression orr FeatureGroup or FieldValuesClause?
    # Empty value list matches all docs in field?
    # Empty field matches all docs?
    """
    The basic building block for querying and retrieving documents by their contents.

    """

    field: str
    values: dc.field(default_factory=list)

    def to_html(self, idx: HyperrealIndex):
        pass
