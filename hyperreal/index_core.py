"""

Main interface for working with an index: the HyperrealIndex class itself.


Examples, setup, usage, general ideas.

Interactions between the schema and this?

How does a query work?

Documentation page: querying.

"""

import collections
import concurrent.futures as cf
from typing import Any, Callable, Iterable, Optional

from pyroaring import BitMap, BitMap64

from . import corpus, db_utilities, index_builder, value_handlers


class Migration:

    def __init__(
        self,
        from_version: Optional[str],
        to_version: str,
        steps: list[str | Callable],
    ):
        self.from_version = from_version
        self.to_version = to_version
        self.steps = steps
        self.validate_steps()

    def validate_steps(self):
        invalid_steps = [
            step for step in self.steps if not (isinstance(step, str) or callable(step))
        ]
        if invalid_steps:
            raise ValueError(
                "Steps must be either strs for SQL queries, or callable functions. "
                f"These steps are invalid: {invalid_steps}"
            )


class IndexPlugin:

    plugin_name: str
    migrations: list[Migration]
    current_version: str

    def __init__(
        self,
        plugin_name: str,
        current_version: Optional[str] = None,
        migrations: Optional[list[Migration]] = None,
    ):

        self.plugin_name = plugin_name
        self.current_version = current_version
        self.migrations = migrations or []

        self.setup_validate_migrations()

    def setup_validate_migrations(self):
        """
        Validate structure of migrations.

        A valid set of migrations needs to:

            - Either be empty, indicating no state is maintained (stateless plugin), or
              have a migration to the `current_version`.
            - Have only one migration per `from_version`. This is necessary to ensure
              that we always follow a consistent pathway through the migrations.

        """

        # An empty set of migrations is valid, and is treated as a stateless migration.
        if not self.migrations:
            self.version_map = {}
            return

        # Validate
        version_map = collections.defaultdict(list)
        for m in self.migrations:
            version_map[m.from_version].append(m)

        invalid_migrations = {
            version: migrations
            for version, migrations in version_map.items()
            if len(migrations) > 1
        }

        if invalid_migrations:
            raise ValueError(
                "The following migrations are invalid as they share the same "
                f"from_version: {invalid_migrations=}"
            )

        # Check that we can migrate to the current version. If no migrations are
        # specified we treat this as a stateless migration.
        to_versions = set(m.to_version for m in self.migrations)

        if self.current_version not in to_versions:
            raise ValueError(
                f"There is no migration to the current version {current_version=}."
            )

        self.version_migration_map = {v: m[0] for v, m in version_map.items()}


core_migration = Migration(
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
)
core_schema = IndexPlugin("hyperreal_core", "1", migrations=[core_migration])


class HyperrealIndex:

    def __init__(
        self,
        index_path,
        corpus: corpus.Corpus,
        pool: Optional[cf.Executor],
        plugins: Optional[list[IndexPlugin]] = None,
    ):
        """Plugins are initialised once at startup."""

        self.index_path = index_path
        self.corpus = corpus
        self.pool = pool
        self.db = db_utilities.connect_sqlite(index_path)
        self.plugins = plugins or []

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

        self.db.execute("begin")
        # TODO: foreign keys or no? Probably no?

        self._ensure_migrated(core_schema)
        self._init_plugins()
        self.db.execute("commit")

    def _init_plugins(self):
        """Initialise plugins, ensuring all migrations have been run."""
        for plugin in self.plugins:
            # Check version against the index.
            # run migrations if necessary.
            self._ensure_migrated(plugin)

    def _apply_migration(self, migration: Migration):

        for i, step in enumerate(migration.steps):
            # TODO: log and raise meaningful error
            if isinstance(step, str):
                self.db.execute(step)
            elif callable(step):
                step(self)

    def _ensure_migrated(self, plugin: IndexPlugin):

        index_version = self.get_plugin_version(plugin.plugin_name)

        while index_version != plugin.current_version:

            migration = plugin.version_migration_map[index_version]
            index_version = migration.to_version
            # TODO: log migrations being applied
            self._apply_migration(migration)
            # TODO: error message when index version not in possible migrations

        self._set_plugin_version(plugin.plugin_name, index_version)

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
            for field, name, cardinality, position_count in field_details:

                handler = self.corpus.name_handlers[name]

                _field_handlers[field] = (
                    handler,
                    cardinality,
                    position_count,
                )

            self._field_handlers = _field_handlers

        return self._field_handlers

    def __getstate__(self):
        return self.index_path, self.corpus

    def __setstate__(self, args):
        """
        Used to handle pickling the index object for use of an index in a process pool.

        Note that the process pool is not available on a reconstituted index.

        """
        self.__init__(args[0], args[1], None)

    def rebuild(self, doc_batch_size: int = 1000, max_workers: Optional[int] = None):
        """(Re)build this index, indexing all documents on the corpus."""

        index_builder.build_index(
            self.index_path, self.corpus, self.pool, doc_batch_size, max_workers
        )

        # The schema might have changed, so invalidate if present.
        if hasattr(self, "_field_handlers"):
            del self._field_handlers

    def doc_ids_to_keys(self, doc_ids: Iterable[int]):
        """Generate document keys on the corpus from the given doc_ids."""
        pass

    def all_doc_ids(self) -> BitMap:
        """Returns all doc_ids in the collection."""
        pass

    def docs(self, doc_ids):
        """Iterate through documents on the corpus matching doc_ids."""
        doc_keys = self.doc_ids_to_keys(doc_ids)
        for (doc_id, doc_key), doc in self.corpus():
            pass

    def __getitem__(self, field, value) -> BitMap:
        """Retrieve documents matching a literal field value."""

        """
        Errors and validation:

        - field needs to be in field handlers, otherwise error
        - missing value is fine as long as the field exists: just return empty docs.
        - slices for ranges, but only on sortable fields. Step parameter must be None.
        
        """

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


class Query:
    # TODO: make this a fragment, and make sure it prints nicely.
    def evaluate(self, idx: HyperrealIndex):
        pass

    def render_into(self, idx: HyperrealIndex):
        pass


class Any(Query):

    def __init__(self, *args):
        self.args = args

        # TODO - validate input arguments in general

    def evaluate(self, idx: HyperrealIndex):

        # TODO - validate input arguments for the index? Or does that get pushed down
        # to the index itself?

        result = BitMap()

        for arg in self.args:

            if isinstance(arg, Query):
                result |= arg.evaluate()

            elif isinstance(arg, tuple):
                result |= idx.or_query(arg)

        return result


class All(Query):

    def __init__(self, *args):
        self.args = args

    def evaluate(self, idx: HyperrealIndex):

        if not self.args:
            return BitMap()

        # Initialise with the first arg
        if isinstance(self.args[0], Query):
            result = self.args[0].evaluate(idx)

        elif isinstance(self.args[0], tuple):
            result = idx.and_query(self.args[0])

        for arg in self.args[1:]:

            if isinstance(arg, Query):
                result &= arg.evaluate(idx)

            elif isinstance(arg, tuple):
                result &= idx.and_query(arg)

        return result


class Not(Query):
    def __init__(self, potentially_keep, remove):
        self.keep = potentially_keep
        self.remove = remove

    def evaluate(self, idx: HyperrealIndex):
        return self.keep.evaluate(idx) - self.remove.evaluate(idx)
