"""
An Index is a boolean inverted index, mapping field, value tuples in documents
to document keys.

"""

# pylint: disable=too-many-lines

import array
import atexit
import collections
import concurrent.futures as cf
import heapq
import logging
import math
import multiprocessing as mp
import os
import random
import sqlite3
import tempfile
from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any, Hashable, Optional, Union

from pyroaring import AbstractBitMap, BitMap

from hyperreal import _index_schema, db_utilities, utilities
from hyperreal.corpus import Corpus

logger = logging.getLogger(__name__)


class CorpusMissingError(AttributeError):
    """
    Raised when using functions that assume a corpus, but no corpus is present.

    """


class UnsupportedCorpusOperation(AttributeError):
    """
    Raised when the corpus provided does not support this function.
    """


class IndexingError(AttributeError):
    "Raised for specific problems during indexing."


FeatureKey = tuple[str, Hashable]
FeatureKeyOrId = Union[FeatureKey, int]
FeatureIdAndKey = tuple[int, str, Hashable]
BitSlice = list[BitMap]


def atomic(writes=False):
    """
    Wrap the decorated interaction with SQLite in a transaction or savepoint.

    Uses savepoints - if no enclosing transaction is present, this will create
    one, if a transaction is in progress, this will be nested as a non durable
    savepoint within that transaction.

    By default, transactions are considered readonly - set this to false to
    mark when changes happen so that housekeeping functions can run at the
    end of a transaction.

    """

    def atomic_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            try:
                self._transaction_level += 1
                self.db.execute(f'savepoint "{func.__name__}"')

                results = func(*args, **kwargs)

                if writes:
                    self._changed = True

                return results

            except Exception:
                self.logger.exception("Error executing index method.")
                # Rewind to the previous savepoint, then release it
                # This is necessary to behave nicely whether we are operating
                # inside a larger transaction or just in autocommit mode.
                self.db.execute(f'rollback to "{func.__name__}"')
                raise

            finally:
                self._transaction_level -= 1

                # We've decremented to the final transaction level and are about
                # to commit.
                if self._transaction_level == 0 and self._changed:
                    self.logger.info("Changes detected - updating clusters.")
                    # Note that this will check for changed queries, and will
                    # therefore be a noop if there aren't any.
                    self._update_changed_clusters()
                    self._changed = False

                self.db.execute(f'release "{func.__name__}"')

        return wrapper

    return atomic_wrapper


class Index:
    """
    An index represents access to a collection of documents defined by a corpus.

    """

    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        db_path,
        corpus: Corpus,
        pool=None,
        random_seed=None,
    ):
        """
        The corpus object is optional - if not provided certain operations such
        as retrieving or rendering documents won't be possible.

        A concurrent.futures pool may be provided to control concurrency
        across different operations. If not provided, a pool will be initialised
        using within a `spawn` mpcontext.

        Note that the index is structured so that db_path is the only necessary
        state, and can always be reinitialised from just that path.

        A random seed can be specified - this will be used with the standard
        library's random module to fix the seed state + enable some kinds of
        reproducibility. Note that this isn't guaranteed to be consistent
        across Python versions.

        """
        self.db_path = db_path
        self.db = db_utilities.connect_sqlite(self.db_path)
        self.random = random.Random(random_seed)

        # _created_pool indicates that we need to cleanup the pool.
        self._created_pool = False
        self._pool = pool

        for statement in """
            pragma synchronous=NORMAL;
            pragma foreign_keys=ON;
            pragma journal_mode=WAL;
            """.split(
            ";"
        ):
            self.db.execute(statement)

        migrated = _index_schema.migrate(self.db)

        if migrated:
            self.db.execute("begin")
            self._update_changed_clusters()
            self.db.execute("commit")

        self.corpus = corpus
        self.field_values = corpus.field_values

        # For tracking the state of nested transactions. This is incremented
        # everytime a savepoint is entered with the @atomic() decorator, and
        # decremented on leaving. Housekeeping functions will run when leaving
        # the last savepoint by committing a transaction.
        self._transaction_level = 0
        self._changed = False

        # Set up a context specific adapter for this index.
        self.logger = logging.LoggerAdapter(logger, {"index_db_path": self.db_path})

    @property
    def pool(self):
        """
        Lazily initialised multiprocessing pool if none is provided on init.

        Note that if a pool is generated on demand an atexit handler will be created
        to cleanup the pool and pending tasks. If a pool is passed in to this instance,
        no cleanup action will be taken.

        """
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(mp_context=mp.get_context("spawn"))
            self._created_pool = True

            def shutdown_pool(idx):
                "Create an exit handler to ensure that the pool is cleaned up on exit."
                if idx._pool is not None:
                    idx._pool.shutdown(wait=False, cancel_futures=True)

            atexit.register(shutdown_pool, self)

        return self._pool

    @classmethod
    def is_index_db(cls, db_path):
        """Returns True if a db exists at db_path and is an index db."""
        try:
            db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            return (
                list(db.execute("pragma application_id"))[0][0]
                == _index_schema.MAGIC_APPLICATION_ID
            )
        except sqlite3.OperationalError:
            return False

    def __enter__(self):
        self.db.execute("begin")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getstate__(self):
        return self.db_path, self.corpus

    def __setstate__(self, args):
        self.__init__(args[0], corpus=args[1])

    def close(self):
        """
        Close the resources associated with this index.

        This includes the database holding the index, the corpus if provided,
        and the multiprocessing pool (if one was created rather than passed
        in).

        """
        if self._created_pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

        self.db.close()
        self.corpus.close()

    @atomic()
    def __getitem__(self, key: FeatureKeyOrId) -> BitMap:
        """
        Retrieve the set of documents matching a literal feature from the index.

        A feature is represented as either a integer `feature_id` or a tuple
        of `("field_name", value)`.

        Note that an exception will be raised if the field doesn't exist, but not
        if the value doesn't exist on a valid field. If the value doesn't exist on a
        field the return result indicates no matching documents.

        """

        if isinstance(key, int):
            try:
                return list(
                    self.db.execute(
                        "select doc_ids from inverted_index where feature_id = ?",
                        [key],
                    )
                )[0][0]
            except IndexError:
                return BitMap()

        elif isinstance(key, tuple):
            try:
                field, value = key

                if field not in self.field_values:
                    raise KeyError(f"Field {field} does not exist on this index.")

                indexed_value = self.field_values[field].to_index(value)

                return list(
                    self.db.execute(
                        """
                        select doc_ids 
                        from inverted_index 
                        where (field, value) = (?, ?)
                        """,
                        (field, indexed_value),
                    )
                )[0][0]
            except IndexError:
                return BitMap()

        else:
            raise ValueError(
                "Must provide an integer feature_id or a ('field', value) pair."
            )

    def lookup_feature(self, feature_id: int) -> tuple[str, Any]:
        """Lookup the (field, value) for this feature by feature_id."""

        results = list(
            self.db.execute(
                "select field, value from inverted_index where feature_id = ?",
                [feature_id],
            )
        )

        if results:
            field, value = results[0]
            return field, self.field_values[field].from_index(value)

        raise KeyError(f"Feature with id '{feature_id}' not found.")

    def lookup_feature_id(self, key: tuple[str, Any]) -> int:
        """Lookup the feature_id for this feature"""

        field, value = key
        indexed_value = self.field_values[field].to_index(value)
        results = list(
            self.db.execute(
                "select feature_id from inverted_index where (field, value) = (?, ?)",
                (field, indexed_value),
            )
        )

        if results:
            return results[0][0]

        raise KeyError(f"Feature with key '{key}' not found.")

    def rebuild(
        self,
        doc_batch_size=1000,
        working_dir=None,
        workers=None,
        index_positions=False,
    ):
        """
        Rebuilds the index from the corpus.

        This method will index the entire corpus from scratch. If the corpus has
        already been indexed, it will be atomically replaced.

        By default, a temporary directory will be created to store temporary
        files which will be cleaned up at the end of the process. If
        `working_dir` is provided, temporary files will be stored there and
        cleaned up as processing continues, but the directory itself won't be
        cleaned up at the end of the process.

        Implementation notes:

        - aims to only load and process small batches in parallel in the worker
          threads: documents will be streamed through so that memory is used
          only for storing the incremental index results
        - limits the number of batches in flight at the same time
        - incrementally merges background batches to a single file
        - new index content is created in the background, and indexed content is
          written to the index in the background.

        """

        # pylint: disable=too-many-statements

        workers = workers or self.pool._max_workers

        try:
            self.db.execute("pragma foreign_keys=0")
            self.db.execute("begin")
        except sqlite3.OperationalError as exc:
            raise IndexingError(
                "The `index` method can't be called in a nested transaction."
            ) from exc

        try:
            manager = mp.Manager()
            write_lock = manager.Lock()

            detach = False

            # pylint: disable-next=consider-using-with
            tempdir = working_dir or tempfile.TemporaryDirectory()
            temp_index = os.path.join(tempdir.name, "temp_index.db")

            # We're still inside a transaction here, so processes reading from
            # the index won't see any of these changes until the release at
            # the end.
            self.db.execute("delete from doc_key")

            # we will associate document keys to internal document ids
            # sequentially
            keys = enumerate(self.corpus.keys())

            batch_key_id_map = {}
            batch_size = 0

            futures = set()

            self.logger.info("Beginning indexing.")

            for key in keys:
                self.db.execute("insert into doc_key values(?, ?)", key)
                batch_key_id_map[key[1]] = key[0]
                batch_size += 1

                if batch_size >= doc_batch_size:
                    self.logger.debug("Dispatching batch for indexing.")

                    # Dispatch the batch
                    futures.add(
                        self.pool.submit(
                            _index_docs,
                            self.corpus,
                            batch_key_id_map,
                            temp_index,
                            index_positions,
                            write_lock,
                        )
                    )
                    batch_key_id_map = {}
                    batch_size = 0

                    # Be polite and avoid filling up the queue.
                    if len(futures) >= workers + 1:
                        done, futures = cf.wait(futures, return_when="FIRST_COMPLETED")

                        for f in done:
                            f.result()

            # Dispatch the final batch.
            if batch_key_id_map:
                self.logger.debug("Dispatching final batch for indexing.")
                futures.add(
                    self.pool.submit(
                        _index_docs,
                        self.corpus,
                        batch_key_id_map,
                        temp_index,
                        index_positions,
                        write_lock,
                    )
                )

            self.logger.info("Waiting for batches to complete.")

            # Zero out existing features, but don't reassign them
            self.db.execute(
                """
                update inverted_index set
                    docs_count = 0,
                    doc_ids = ?1
                """,
                [BitMap()],
            )

            # Zero out positional information
            self.db.execute("delete from position_doc_map")
            self.db.execute("delete from position_index")

            # Make sure all of the batches have completed.
            for f in cf.as_completed(futures):
                f.result()

            self.logger.info("Batches complete - merging into main index.")

            # Now merge back to the original index, preserving feature_ids
            # if this is a reindex operation.
            self.db.execute("attach ? as tempindex", [temp_index])
            detach = True

            self.db.execute(
                """
                create index tempindex.field_value on inverted_index_segment(
                    field, value, first_doc_id
                )
                """
            )

            # Actually populate the new values - inverted index
            self.db.execute(
                """
                replace into inverted_index(feature_id, field, value, docs_count, doc_ids)
                    select
                        (
                            select feature_id
                            from inverted_index ii
                            where (ii.field, ii.value) = (iis.field, iis.value)
                        ) as feature_id,
                        field,
                        value,
                        sum(docs_count) as docs_count,
                        roaring_union(doc_ids) as doc_ids
                    from inverted_index_segment iis
                    group by field, value
                    -- Order is an insert optimisation and not strictly necessary
                    order by feature_id, field, value
                """
            )

            # Position index
            self.db.execute(
                """
                insert into position_index(feature_id, first_doc_id, position_count, positions)
                    select
                        (
                            select feature_id
                            from inverted_index ii
                            where (ii.field, ii.value) = (iis.field, iis.value)
                        ) as feature_id,
                        first_doc_id,
                        position_count,
                        positions
                    from inverted_index_segment iis
                    where position_count > 0
                    -- Order is an insert optimisation and not strictly necessary
                    order by feature_id, first_doc_id
                """
            )

            self.db.execute(
                """
                insert into position_doc_map
                    select *
                    from batch_position
                """
            )

            # Update docs_counts in the clusters
            self.db.execute(
                """
                update feature_cluster set
                    docs_count = (
                        select docs_count
                        from inverted_index ii
                        where ii.feature_id = feature_cluster.feature_id
                    )
                """
            )

            # Write the field summary
            self.db.execute("delete from field_summary")
            self.db.execute(
                """
                insert into field_summary
                select
                    field,
                    count(*) as distinct_values,
                    min(value) as min_value,
                    max(value) as max_value,
                    coalesce(
                        (
                            select sum(position_count)
                            from position_index
                            inner join inverted_index using(feature_id)
                            where field = ii.field
                        ),
                        0
                    )
                from inverted_index ii
                group by field
                """
            )

            # Update all cluster stats based on new index stats
            self.db.execute(
                "insert into changed_cluster select cluster_id from cluster"
            )

            self.db.execute("commit")

            self.db.execute("begin")
            self._update_changed_clusters()
            self.db.execute("commit")

        except Exception:
            self.logger.exception("Indexing failure.")
            self.db.execute("rollback")
            raise

        finally:
            self.db.execute("pragma foreign_keys=1")
            manager.shutdown()

            if detach:
                self.db.execute("detach tempindex")

            tempdir.cleanup()

        self.logger.info("Indexing completed.")

    @atomic()
    def convert_query_to_keys(self, query) -> dict[Hashable, int]:
        """
        Return a mapping of keys to their doc_id.

        This can be passed directly to corpus objects to retrieve matching
        docs.

        """

        key_docs = {}
        for doc_id in query:
            doc_key = list(
                self.db.execute(
                    "select doc_key from doc_key where doc_id = ?", [doc_id]
                )
            )[0][0]
            key_docs[doc_key] = doc_id

        return key_docs

    @atomic()
    def iter_field_docs(self, field, min_docs=1):
        """
        Iterate through all values and corresponding doc_ids for the given field.

        Iteration is by lexicographical order of the values.

        """
        if field not in self.field_values:
            raise KeyError(
                f"Field {field} is not defined on this index. "
                f"Valid fields are {self.field_values.keys()}"
            )

        value_docs = (
            (self.field_values[field].from_index(row[0]), *row[1:])
            for row in self.db.execute(
                """
                select value, docs_count, doc_ids
                from inverted_index
                where field = ?1
                    and docs_count >= ?2
                order by value
                """,
                [field, min_docs],
            )
        )

        yield from value_docs

    @atomic()
    def intersect_queries_with_field(
        self, queries: dict[Hashable, AbstractBitMap], field: str
    ) -> tuple[list[Any], list[int], dict[Hashable, list[int]]]:
        """
        Intersect all the given queries with all values in the chosen field.

        Note that this can take a long time with fields with many values, such
        as tokenised text. This is best used with single value fields of low
        cardinality (<1000 distinct values). Examples of this might be
        datetimes truncated to a month, or ordinal ranges such as a likert
        scale.

        """

        intersections = collections.defaultdict(list)
        values = []
        totals = []

        for value, docs_count, doc_ids in self.iter_field_docs(field):
            values.append(value)
            totals.append(docs_count)

            for name, query in queries.items():
                inter = query.intersection_cardinality(doc_ids)
                intersections[name].append(inter)

        return values, totals, intersections

    def docs(self, query):
        """Retrieve the documents matching the given query set."""
        keys = self.convert_query_to_keys(query)
        for key, doc in self.corpus.docs(sorted(keys)):
            yield keys[key], key, doc

    def sample_bitmap(self, bitmap, random_sample_size):
        """
        Sample up to random_sample_size members from bitmap.

        If there are fewer than random_sample_size members in the bitmap, return
        a copy of the bitmap.

        Uses the current state of the indexes random number generator to
        enable repeatable runs.

        """

        b = len(bitmap)
        if b > random_sample_size:
            sampled = BitMap(
                bitmap[i] for i in self.random.sample(range(b), random_sample_size)
            )
            return sampled

        return bitmap.copy()

    @staticmethod
    def match_doc_features(
        doc_features, features_to_match
    ) -> dict[str, dict[Any, list]]:
        """
        Identify which parts of `features` occur in this `doc_features`.

        This returns a mapping showing the fields and associated values that match the
        doc from the given list of features. If the field is a positional/sequence
        field it will also return the numerical offset of that value in the field to
        allow the production of concordances, snippets or passages.

        This can be used with a set of features constituting a query and the
        `doc_features` method to annotate a set of results with appropriate context
        to show why this document was retrieved for this query.

        """
        matches = collections.defaultdict(lambda: collections.defaultdict(list))

        for field, match_values in features_to_match.items():
            if field not in doc_features:
                continue

            doc_values = doc_features[field]

            if isinstance(doc_values, list):
                for position, value in enumerate(doc_values):
                    if value in match_values:
                        matches[field][value].append(position)

            elif isinstance(doc_values, set):
                for value in doc_values & match_values:
                    matches[field][value] = []

            else:
                if doc_values in match_values:
                    matches[field][doc_values] = []

        return matches

    @atomic()
    def structured_doc_sample(self, docs_per_cluster=100, cluster_ids=None):
        """
        Create a sample of documents, using the current clustering as a sampling
        structure.

        By default 100 documents will be sampled from each cluster - sampling can be
        disabled by setting docs_per_cluster to 0.

        Optionally specify specific clusters to sample from using `cluster_ids`,
        otherwise all clusters will be sampled.

        Documents will be sampled from clusters in order of increasing frequency, and
        will be sampled without replacement.

        Will return a mapping of cluster_ids to sampled documents, and also a map of all
        clusters for each of those documents.

        """
        all_clusters = self.top_cluster_features(top_k=0)
        cluster_order = reversed(all_clusters)

        cluster_ids = set(cluster_ids or self.cluster_ids)

        already_sampled = BitMap()
        cluster_samples = {}

        # Per cluster samples
        for cluster_id, _, _ in cluster_order:
            if cluster_id in cluster_ids:
                cluster_docs = self.cluster_docs(cluster_id) - already_sampled
                if docs_per_cluster > 0:
                    sample = self.sample_bitmap(cluster_docs, docs_per_cluster)
                else:
                    sample = cluster_docs
                cluster_samples[cluster_id] = sample
                already_sampled |= sample

        # Clusters for all of the sampled documents.
        sample_clusters = {
            cluster_id: c
            for cluster_id, _, _ in all_clusters
            if (c := already_sampled & self.cluster_docs(cluster_id))
        }

        return cluster_samples, sample_clusters

    def indexed_field_summary(self):
        """
        Return a summary tables of the indexed fields.

        """
        return list(self.db.execute("select * from field_summary"))

    @atomic(writes=True)
    def initialise_clusters(self, n_clusters, min_docs=1, include_fields=None):
        """
        Initialise the model with the given number of clusters.

        Features that retrieve at least `min_docs` are randomly assigned to
        one of the given clusters.

        `include_fields` can be specified to limit initialising the model to features
        from only the selected fields.

        """

        # Note - foreign key constraints handle most of the associated metadata,
        # we just do one extra step to avoid a circular trigger
        self.db.execute("delete from feature_cluster")
        self.db.execute("delete from cluster")

        self.db.execute("create temporary table if not exists include_field(field)")
        self.db.execute("delete from include_field")

        if include_fields:
            self.db.executemany(
                "insert into include_field values(?)", [[f] for f in include_fields]
            )
        else:
            self.db.execute("insert into include_field select field from field_summary")

        feature_ids = list(
            self.db.execute(
                """
                select
                    feature_id,
                    docs_count
                from inverted_index
                inner join include_field using(field)
                where docs_count >= ?
                -- Note: specify the ordering to ensure reproducibility, as these
                -- results will be shuffled.
                order by feature_id
                """,
                [min_docs],
            )
        )

        self.random.shuffle(feature_ids)

        clusters = ((i, feature_ids[i::n_clusters]) for i in range(n_clusters))

        self.db.executemany(
            """
            insert into feature_cluster(cluster_id, feature_id, docs_count)
                values(?, ?, ?)
            """,
            (
                (cluster_id, *feature)
                for cluster_id, features in clusters
                for feature in features
            ),
        )

        self.db.execute("drop table include_field")

        self.logger.info(f"Initialised new model with {n_clusters} clusters.")

    @atomic(writes=True)
    def delete_clusters(self, cluster_ids):
        """Delete the specified clusters."""
        # The cluster table will be automatically updated by the housekeeping functionality
        self.db.executemany(
            "delete from feature_cluster where cluster_id = ?",
            [[c] for c in cluster_ids],
        )

        self.logger.info(f"Deleted clusters {cluster_ids}.")

    @atomic(writes=True)
    def merge_clusters(self, cluster_ids):
        """Merge all clusters into the first cluster_id in the provided list."""

        merge_cluster_id = cluster_ids[0]

        for cluster_id in cluster_ids[1:]:
            self.db.execute(
                "update feature_cluster set cluster_id=? where cluster_id=?",
                [merge_cluster_id, cluster_id],
            )

        self.logger.info(f"Merged {cluster_ids} into {merge_cluster_id}.")

        return merge_cluster_id

    @atomic(writes=True)
    def delete_features(self, feature_ids):
        """Delete the given features from the model."""
        self.db.executemany(
            "delete from feature_cluster where feature_id=?",
            [[f] for f in feature_ids],
        )

        self.logger.info(f"Delete features {feature_ids}.")

    def next_cluster_id(self):
        """Return a new cluster_id, higher than anything already assigned."""
        next_cluster_id = list(
            self.db.execute(
                """
                select
                    coalesce(max(cluster_id), 0) + 1
                from cluster
                """
            )
        )[0][0]

        return next_cluster_id

    @atomic(writes=True)
    def create_cluster_from_features(self, feature_ids):
        """
        Create a new cluster from the provided set of features.

        The features must exist in the index.

        """

        next_cluster_id = self.next_cluster_id()

        self.db.executemany(
            """
            insert or ignore into feature_cluster(feature_id, cluster_id, docs_count)
                values (
                    ?1,
                    ?2,
                    (
                        select docs_count from inverted_index where feature_id = ?1
                    )
                )
            """,
            [(f, next_cluster_id) for f in feature_ids],
        )

        self.db.executemany(
            "update feature_cluster set cluster_id = ? where feature_id = ?",
            [(next_cluster_id, f) for f in feature_ids],
        )

        self.logger.info(f"Created cluster {next_cluster_id} from {feature_ids}.")

        return next_cluster_id

    @atomic(writes=True)
    def pin_features(self, feature_ids: Sequence[int], pinned: bool = True):
        """
        Pin (or unpin) the given features.

        Pinning a feature prevents it from being moved during the clustering
        feature. This can be used to preserve interesting combinations of
        features together in the same cluster.

        """

        self.db.executemany(
            "update feature_cluster set pinned = ? where feature_id = ?",
            ((pinned, f) for f in feature_ids),
        )

    @atomic(writes=True)
    def pin_clusters(self, cluster_ids: Sequence[int], pinned: bool = True):
        """
        Pin (or unpin) the given clusters.

        A pinned cluster will not be modified by the automated clustering
        routine. This can be used to preserve useful clusters and allow
        remaining unpinned clusters to be refined further.
        """
        self.db.executemany(
            "update cluster set pinned = ? where cluster_id = ?",
            ((pinned, c) for c in cluster_ids),
        )

    @property
    def cluster_ids(self):
        """The ids of all defined feature-clusters."""
        return [
            r[0]
            for r in self.db.execute(
                "select cluster_id from cluster order by cluster_id"
            )
        ]

    @property
    def pinned_cluster_ids(self):
        """The ids of all clusters pinned to prevent further change."""
        return {
            r[0]
            for r in self.db.execute(
                "select cluster_id from cluster where pinned order by cluster_id"
            )
        }

    @atomic()
    def top_cluster_features(self, top_k=20):
        """Return the top_k features according to the number of matching documents."""

        cluster_docs = self.db.execute(
            """
            select cluster_id, docs_count
            from cluster
            order by docs_count desc
            """
        )

        clusters = [
            (cluster_id, docs_count, self.cluster_features(cluster_id, top_k=top_k))
            for cluster_id, docs_count in cluster_docs
        ]

        return clusters

    @atomic()
    def pivot_clusters_by_query(
        self, query, cluster_ids=None, top_k=20, scoring="jaccard"
    ):
        """
        Sort all clusters and features within clusters by similarity with the probe query.

        This function is optimised to yield top ranking results as early as possible,
        to enable streaming outputs as soon as they're ready, such as in the web interface.

        Returns:

            Generator of clusters and features sorted by similarity with the
            associated query.

        Args:
            query: the query object as a bitmap of document IDs
            cluster_ids: an optional sequence
            top_k: the number of top features to return in each cluster
            scoring: The similarity scoring function, currently only "jaccard"
                is supported.

        """

        cluster_ids = cluster_ids or [
            r[0]
            for r in self.db.execute(
                "select cluster_id from cluster order by feature_count desc"
            )
        ]

        jobs = self.pool._max_workers * 2
        futures = [
            self.pool.submit(
                _calculate_query_cluster_cooccurrence,
                self,
                0,
                query,
                cluster_ids[i::jobs],
            )
            for i in range(jobs)
        ]

        weights = []

        for f in cf.as_completed(futures):
            weights.extend(f.result()[1])

        process_order = sorted(weights, key=lambda x: x[1], reverse=True)

        if scoring == "jaccard":
            futures = [
                self.pool.submit(
                    _pivot_cluster_features_by_query_jaccard,
                    self,
                    query,
                    cluster_id,
                    top_k,
                    inter,
                )
                for cluster_id, _, inter in process_order
            ]

        else:
            raise ValueError(
                f"{scoring} method is not supported. "
                "Only jaccard is currently supported."
            )

        return (future.result() for future in futures)

    def cluster_features(self, cluster_id, top_k=2**62):
        """
        Returns an impact ordered list of features for the given cluster.

        If top_k is specified, only the top_k most frequent features by
        document count are returned in descending order.

        """
        cluster_features = [
            (*row[:2], self.field_values[row[1]].from_index(row[2]), row[3])
            for row in self.db.execute(
                """
                select
                    feature_id,
                    field,
                    value,
                    -- Note that docs_count is denormalised to allow
                    -- a per cluster sorting of document count.
                    fc.docs_count
                from feature_cluster fc
                inner join inverted_index ii using(feature_id)
                where cluster_id = ?
                order by fc.docs_count desc
                limit ?
                """,
                [cluster_id, top_k],
            )
        ]

        return cluster_features

    @atomic()
    def union_bitslice(self, features: Sequence[FeatureKeyOrId]):
        """
        Return matching documents and accumulated bitslice for the given set
        of features.

        """

        bitmaps = (self[feature] for feature in features)
        return utilities.compute_bitslice(bitmaps)

    @atomic()
    def cluster_query(self, cluster_id):
        """
        Return matching documents and accumulated bitslice for cluster_id.

        If you only need the matching documents, the `cluster_docs` method is
        faster as it retrieves a precomputed set of documents.

        The matching documents are the documents that contain any terms from
        the cluster. The returned bitslice represents the accumulation of
        features matching across all features and can be used for ranking
        with `utilities.bstm`.

        """

        feature_ids = [r[0] for r in self.cluster_features(cluster_id)]

        return self.union_bitslice(feature_ids)

    def cluster_docs(self, cluster_id: int) -> AbstractBitMap:
        """Return the bitmap of documents covered by this cluster."""
        return list(
            self.db.execute(
                "select doc_ids from cluster where cluster_id = ?", [cluster_id]
            )
        )[0][0]

    def _score_proposed_moves(
        self,
        cluster_feature: dict[Hashable, set[int]],
        cluster_check_feature: dict[Hashable, set[int]],
        probe_query: Optional[AbstractBitMap] = None,
        top_k: int = 1,
    ) -> dict[int, tuple[float, Hashable]]:
        """
        Return the estimated best cluster/s for each feature.

        This should be used in conjunction with the scores output from
        `_measure_feature_cluster_contributions`, which describes the contribution
        of each feature to its own cluster.

        The input is two mappings:

        cluster_feature - the current cluster: features mapping
        cluster_check_features - mapping of cluster: features to check
            against moving into this cluster. Note that features will
            be removed from this set if they are already in this cluster
            in cluster_feature.

        top_k: the number of best scores to keep - the default is to only
            the single best scoring feature.

        """

        # preflight check:
        missing_clusters = {
            cluster_key
            for cluster_key in cluster_check_feature
            if cluster_key not in cluster_feature
        }

        if missing_clusters:
            raise KeyError(
                f"`cluster_feature` is missing clusters with keys: {missing_clusters}"
            )

        futures = [
            self.pool.submit(
                measure_feature_contribution_to_cluster,
                self,
                cluster_key,
                cluster_feature[cluster_key],
                cluster_check_feature[cluster_key] - cluster_feature[cluster_key],
                probe_query,
            )
            # dispatch in sort order for reproducibility with randomisation
            for cluster_key in sorted(cluster_check_feature)
        ]

        best_clusters = collections.defaultdict(list)

        cluster_objectives = {}

        for future in futures:
            result = future.result()
            test_cluster, objective = result[:2]
            cluster_objectives[test_cluster] = objective

            for feature_array, delta_array in [result[2:4], result[4:6]]:
                for feature, delta in zip(feature_array, delta_array):
                    if len(best_clusters[feature]) < top_k:
                        heapq.heappush(best_clusters[feature], (delta, test_cluster))
                    else:
                        heapq.heappushpop(best_clusters[feature], (delta, test_cluster))

        return cluster_objectives, {
            key: sorted(feature_scores, reverse=True)
            for key, feature_scores in best_clusters.items()
        }

    def _find_best_moves(
        self,
        cluster_feature: dict[int, set[int]],
        movable_features=None,
        group_test_top_k=1,
        group_test_batches=None,
        probe_query=None,
        top_k=1,
    ):
        """
        Find the approximate next nearest cluster for each feature in this clustering.
        """

        cluster_ids = list(cluster_feature.keys())
        self.random.shuffle(cluster_ids)

        movable_features = movable_features or set.union(*cluster_feature.values())

        if group_test_batches and group_test_batches > len(cluster_ids) / 2:
            group_test_batches = None
            self.logger.info(
                f"{group_test_batches=} is too high to be useful for {len(cluster_ids)=}, "
                "disabling."
            )

        # Group testing, generating randomised groupings of clusters to prune
        # the search space early.
        if group_test_batches:
            cluster_groups = [
                set(cluster_ids[i::group_test_batches])
                for i in range(group_test_batches)
            ]

            group_features = {}
            group_feature_checks = {}

            for group in cluster_groups:
                group_key = tuple(sorted(group))
                this_group_features = set.union(
                    *(cluster_feature[key] for key in group_key)
                )

                group_features[group_key] = this_group_features
                group_feature_checks[group_key] = movable_features

            _, best_groups = self._score_proposed_moves(
                group_features,
                group_feature_checks,
                probe_query=probe_query,
                top_k=group_test_top_k,
            )

            cluster_feature_checks = collections.defaultdict(set)

            # Convert best group results into individual cluster checks
            for feature, groups in best_groups.items():
                if feature in movable_features:
                    for _, group_key in groups:
                        for cluster_key in group_key:
                            cluster_feature_checks[cluster_key].add(feature)

        # The dense testing case is much much simpler, but also much slower!
        else:
            cluster_feature_checks = {c: movable_features for c in cluster_ids}

        return self._score_proposed_moves(
            cluster_feature,
            cluster_feature_checks,
            probe_query=probe_query,
            top_k=top_k,
        )

    def _refine_feature_groups(
        self,
        cluster_feature: dict[int, set[int]],
        iterations: int = 10,
        minimum_cluster_features: int = 1,
        pinned_features: Optional[Iterable[int]] = None,
        probe_query: Optional[AbstractBitMap] = None,
        target_clusters: Optional[int] = None,
        tolerance: float = 0.05,
        group_test_batches: Optional[int] = None,
        group_test_top_k: int = 2,
    ) -> tuple[dict[int, set[int]], set[int]]:
        """
        Low level function for iteratively refining a feature clustering.

        cluster_features is a mapping from a cluster_key to a set of feature_ids.

        This is most useful if you want to explore specific clustering
        approaches without the constraint of the saved clusters.

        To change the number of clusters in the model, set target clusters to
        a different number of clusters than in cluster_feature - the clustering
        will be adjusted to the target. The newly generated IDs will be returned
        along with the new feature clustering.

        Pinned features will not be considered as candidates for moving.

        If a probe_query is provided, it will be intersected with all features
        for clustering: this is used to generate a clustering for a subset of
        the data. Note that features may not intersect with the probe query -
        clustering is not well defined in this case and should be used with
        care.

        target_clusters: specifies a target number of clusters for the model: if
        there are more clusters, the clusters that contribute least to the
        objective will be dissolved. Dissolves will be conducted evenly per
        iteration, rather than all at once. If there are fewer clusters than
        target, new clusters will be created by sampling from the largest
        cluster at random.

        tolerance: specifies a termination tolerance. If fewer than
        tolerance * total_features features move in an iteration, terminate
        early. The default is set at 0.05 - the model is considered
        converged if less than 5% of the features have moved during an
        iteration.

        top_k: the number of nearest neighbour clusters to consider as move
        candidates.
        """

        # pylint: disable=too-many-branches,too-many-statements

        target_clusters = target_clusters or len(cluster_feature)

        # Handle cases where moves are not possible by returning immediately.
        # If cluster_feature is empty
        n_clusters = len(cluster_feature.keys())
        if n_clusters == 0:
            return cluster_feature, set()
        # Or if there is only one cluster, and target_clusters is the same
        if n_clusters == 1 == target_clusters:
            return cluster_feature, set()

        # Make sure to copy the input dict
        cluster_feature = {
            cluster_id: features.copy()
            for cluster_id, features in cluster_feature.items()
        }

        feature_cluster = {
            feature_id: cluster_id
            for cluster_id, features in cluster_feature.items()
            for feature_id in features
        }

        pinned_features = set(pinned_features) if pinned_features else set()
        # The set of clusters with pinned features - these will be used to
        # avoid interfering with pinned features when dividing or dissolving
        # clusters.
        clusters_with_pinned_features = {
            cluster_id
            for cluster_id, features in cluster_feature.items()
            if features & pinned_features
        }
        movable_features = {
            f
            for features in cluster_feature.values()
            for f in features
            if f not in pinned_features
        }

        movable_feature_list = list(movable_features)

        changed_clusters = set(cluster_feature)

        # Generate new cluster_ids and empty clusters if we have less clusters
        # than target_clusters. New clusters are formed by splitting the
        # current largest cluster by number of features roughly in half.
        next_cluster_id = max(cluster_feature) + 1
        new_clusters = target_clusters - len(cluster_feature)
        new_cluster_ids = list(range(next_cluster_id, next_cluster_id + new_clusters))

        for new_cluster_id in new_cluster_ids:
            _, largest_cluster, largest_cluster_features = max(
                (len(features), cluster_id, features)
                for cluster_id, features in cluster_feature.items()
            )
            split = {
                feature
                for feature in largest_cluster_features
                if self.random.random() < 0.5 and feature not in pinned_features
            }
            cluster_feature[new_cluster_id] = split
            cluster_feature[largest_cluster] -= split
            for feature_id in split:
                feature_cluster[feature_id] = new_cluster_id

        assigned_cluster_ids = set(cluster_feature)

        # Work out how many low objective clusters to dissolve on each iteration.
        dissolve_clusters = max(0, len(assigned_cluster_ids) - target_clusters)
        # Note that we try to structure it so the very last iteration does not dissolve
        # anything.
        dissolve_per_iteration = math.ceil(dissolve_clusters / max(1, iterations - 1))
        dissolve_cluster_ids = set()

        for iteration in range(iterations):
            # Calculate possible moves given this clustering
            objectives, best_moves = self._find_best_moves(
                cluster_feature,
                group_test_top_k=group_test_top_k,
                group_test_batches=group_test_batches,
                movable_features=movable_features,
                probe_query=probe_query,
            )

            total_objective = sum(objectives.values())

            self.logger.info(
                f"Iteration {iteration + 1}, current objective: {total_objective}"
            )

            # Dissolve target low objective clusters for this iteration
            if dissolve_clusters:
                n_dissolve = min(dissolve_clusters, dissolve_per_iteration)
                dissolve_clusters -= n_dissolve

                dissolve_cluster_order = sorted(
                    (cluster_obj, cluster_id)
                    for cluster_id, cluster_obj in cluster_feature.items()
                    # Don't dissolve any cluster with a pinned feature
                    if cluster_id not in clusters_with_pinned_features
                )
                dissolve_cluster_ids = set(
                    cluster_id for _, cluster_id in dissolve_cluster_order[:n_dissolve]
                )

                self.logger.info(
                    f"Dissolving {len(dissolve_cluster_ids)} low objective clusters"
                )
            else:
                dissolve_cluster_ids = set()

            possible_clusters = list(set(cluster_feature) - dissolve_cluster_ids)
            possible_moves = 0
            actual_moves = 0
            changed_clusters = set()

            self.random.shuffle(movable_feature_list)

            for feature_id in movable_feature_list:
                _, to_cluster = best_moves[feature_id][0]
                current_cluster = feature_cluster[feature_id]

                to_cluster_len = len(cluster_feature[to_cluster])
                current_cluster_len = len(cluster_feature[current_cluster])

                # Accept moves from clusters with many to few features with high prob,
                # accept moves from clusters with few to many feature with low prob.
                move_acceptance_probability = (current_cluster_len - 1) / (
                    to_cluster_len + current_cluster
                )
                # Note that we're squaring the probability here - this makes moves from
                # smaller to larger clusters much less likely to be accepted, without
                # affecting moves from larger to smaller clusters too much. This makes
                # the algorithm converge slower overall, but prevents the tail of
                # clusters with fewer features from being absorbed into a single large
                # cluster. There might be better ways to address this cluster size
                # desire in the objective, but this will do for now.
                accept = self.random.random() < (move_acceptance_probability**2)

                # Handle dissolving clusters
                if current_cluster in dissolve_cluster_ids:
                    # Always accept a move out of a dissolving cluster
                    accept = True

                    # If we're moving into a dissolving cluster as well move
                    # to a randomly selected cluster.
                    if to_cluster in dissolve_cluster_ids:
                        to_cluster = self.random.choice(possible_clusters)

                # Don't move into a dissolving cluster
                elif to_cluster in dissolve_cluster_ids:
                    continue

                # Not actually a move!
                elif current_cluster == to_cluster:
                    continue

                # Don't move things out of a small cluster.
                elif len(cluster_feature[current_cluster]) == minimum_cluster_features:
                    continue

                # We can make this move, so let's try it
                possible_moves += 1

                if accept:
                    actual_moves += 1

                    cluster_feature[current_cluster].discard(feature_id)
                    cluster_feature[to_cluster].add(feature_id)
                    feature_cluster[feature_id] = to_cluster
                    changed_clusters.add(to_cluster)
                    changed_clusters.add(current_cluster)

            for cluster_id in dissolve_cluster_ids:
                if len(cluster_feature[cluster_id]):
                    raise ValueError("This cluster should have been emptied.")
                del cluster_feature[cluster_id]

            self.logger.info(
                f"Finished iteration {iteration + 1}/{iterations}, changed "
                f"{len(changed_clusters)} clusters, moved {actual_moves} features."
            )

            if not dissolve_cluster_ids:
                if (possible_moves / len(movable_features)) < tolerance:
                    self.logger.info(
                        "Terminating refinement due to small number of feature moves."
                    )
                    break

        return cluster_feature, new_cluster_ids

    @atomic()
    def refine_clusters(
        self,
        iterations: int = 10,
        cluster_ids: Optional[Sequence[int]] = None,
        minimum_cluster_features: int = 1,
        probe_query: Optional[AbstractBitMap] = None,
        target_clusters: Optional[int] = None,
        tolerance: float = 0.05,
        group_test_batches: Optional[int] = None,
        group_test_top_k: int = 2,
    ):
        """
        Refine the feature clusters for the current model.

        Optionally provide a list of specific cluster_ids to refine.

        If target_clusters is larger than the current number of clusters in
        the model, the largest clusters by number of features will be split
        to reach the target. This can be used to split all or some selected
        clusters.

        """

        cluster_ids = set(cluster_ids or self.cluster_ids)

        if iterations < 1:
            raise ValueError(
                f"You must specificy at least one iteration, provided '{iterations}'."
            )

        # Establish forward reverse mappings of features to clusters and vice versa.
        cluster_feature = collections.defaultdict(set)
        pinned_features = set()

        for feature_id, cluster_id, pinned in self.db.execute(
            """
            select
                feature_id,
                cluster_id,
                feature_cluster.pinned
            from feature_cluster
            """
        ):
            if cluster_id in cluster_ids:
                cluster_feature[cluster_id].add(feature_id)
                if pinned:
                    pinned_features.add(feature_id)

        # Set target clusters to the current number of clusters, or the
        # provided value. But we also need to account for pinned clusters in
        # the next step, otherwise this will be the wrong count.
        target_clusters = target_clusters or len(cluster_feature)

        # Remove pinned clusters from refinement, and don't count them towards
        # target clusters.
        pinned_clusters = set(self.pinned_cluster_ids)

        for cluster_id in pinned_clusters:
            if cluster_id in cluster_feature:
                del cluster_feature[cluster_id]
                target_clusters -= 1

        if group_test_batches is None:
            group_test_batches = math.ceil(len(cluster_feature) ** 0.5)

        cluster_feature, new_cluster_ids = self._refine_feature_groups(
            cluster_feature,
            iterations=iterations,
            pinned_features=pinned_features,
            minimum_cluster_features=minimum_cluster_features,
            probe_query=probe_query,
            target_clusters=target_clusters,
            tolerance=tolerance,
            group_test_top_k=group_test_top_k,
            group_test_batches=group_test_batches,
        )

        # Map new_cluster_ids generated to actual globally unique IDs.
        # Make sure to copy these out first, as new_cluster_ids might overlap
        # with the global clustering model!
        new_cluster_feature = {
            cluster_id: cluster_feature[cluster_id] for cluster_id in new_cluster_ids
        }

        next_cluster_id = self.next_cluster_id()

        for cluster_id in new_cluster_ids:
            del cluster_feature[cluster_id]
            cluster_feature[next_cluster_id] = new_cluster_feature[cluster_id]
            next_cluster_id += 1

        # Serialise the actual results of the clustering!
        self._update_cluster_feature(cluster_feature)

    @atomic(writes=True)
    def _update_cluster_feature(self, cluster_feature):
        """
        Update the given cluster: feature mapping.

        Note that this only updates the provided clusters: it does not replace
        the entire state of the model. Also note that this can clobber
        cluster_ids if you're not careful.

        """
        self.db.executemany(
            """
            update feature_cluster set cluster_id = ?1 where feature_id = ?2
            """,
            (
                (cluster_id, feature_id)
                for cluster_id, features in cluster_feature.items()
                for feature_id in features
            ),
        )

    def _update_changed_clusters(self):
        """
        Refresh cluster union queries for changed clusters.

        This is usually called from the `atomic` decorator, or when reindexing.

        It is assumed that this is called inside a transaction.

        """

        # First update the feature counts, and remove empty clusters
        self.db.execute(
            """
            update cluster set feature_count = (
                select count(*)
                from feature_cluster fc
                where fc.cluster_id=cluster.cluster_id
            )
            where cluster_id in (select cluster_id from changed_cluster)
            """
        )

        changed = self.db.execute("select cluster_id from changed_cluster")

        # Then update the union statistics for all of the clusters
        bg_args = (
            (
                self,
                cluster_param[0],
                [
                    row[0]
                    for row in self.db.execute(
                        """
                        select feature_id
                        from feature_cluster
                        where cluster_id = ?
                        """,
                        cluster_param,
                    )
                ],
            )
            for cluster_param in changed
        )

        # Note that the data here may not have been committed yet, so we have
        # to read and pass the feature_ids to the background ourselves.
        for cluster_id, query, weight in self.pool.map(_union_query, bg_args):
            self.db.execute(
                """
                update cluster set (docs_count, weight, doc_ids) = (?, ?, ?)
                where cluster_id = ?
                """,
                (len(query), weight, query, cluster_id),
            )

        self.db.execute("delete from cluster where feature_count = 0")
        self.db.execute("delete from changed_cluster")

    @atomic()
    def _or_partition_positions(self, features):
        # Make sure that all of the features are from the same field.
        fields_covered = set()
        query_features = set()

        for f in features:
            if isinstance(f, int):
                fields_covered.add(self.lookup_feature(f)[0])
                query_features.add(f)
            elif isinstance(f, tuple):
                fields_covered.add(f[0])
                query_features.add(self.lookup_feature_id(f))
            else:
                raise TypeError(f"Unsupported feature {f}.")

    @atomic()
    def field_proximity_query(
        self, field, value_clauses: list[list], window_size: int
    ) -> BitMap():
        """
        Find documents where values co-occur within `window_size` proximity.

        This is useful to create more specific and precise searches for language,
        rather than just co-occurence in whole documents. This is especially helpful
        for collections where whole documents may be long and topically diverse.

        """

        # Convert values to feature_ids
        clause_feature_ids = [
            [self.lookup_feature_id((field, value)) for value in clause]
            for clause in value_clauses
        ]

        futures = set()

        for first_doc_id in self._field_partitions(field):
            futures.add(
                self.pool.submit(
                    _field_proximity_query,
                    self,
                    field,
                    clause_feature_ids,
                    first_doc_id,
                    window_size,
                )
            )

        matching_doc_ids = BitMap()

        for future in cf.as_completed(futures):
            matching_doc_ids |= future.result()

        return matching_doc_ids

    def _field_partitions(self, field):
        """
        Return the list of partitions of positional information for the given field.

        Each partition is identified by the `doc_id` of the first document in
        the partition.

        """
        return [
            r[0]
            for r in self.db.execute(
                """
                select first_doc_id
                from position_doc_map
                where field = ?
                """,
                [field],
            )
        ]

    @property
    def positional_fields(self):
        """Return the names of all fields with positional information in this index."""
        return {
            r[0]
            for r in self.db.execute(
                "select field from field_summary where position_count > 0"
            )
        }

    @atomic()
    def _union_position_query(self, first_doc_id, features):
        positions = BitMap()

        for feature_id in features:
            positions |= self._get_partition_positions(first_doc_id, feature_id)

        return positions

    def _get_partition_positions(self, first_doc_id, feature_id):
        positions = list(
            self.db.execute(
                """
            select
                positions
            from position_index
            where (feature_id, first_doc_id) = (?, ?)
            """,
                [feature_id, first_doc_id],
            )
        )
        if positions:
            return positions[0][0]

        return BitMap()

    def _get_partition_header(self, field, first_doc_id):
        header = list(
            self.db.execute(
                """
                select
                    docs_count,
                    doc_ids,
                    doc_boundaries
                from position_doc_map
                where (field, first_doc_id) = (?, ?)
                """,
                [field, first_doc_id],
            )
        )
        if header:
            return header[0]

        raise ValueError(
            f"No position partition corresponding to {field=}, {first_doc_id=}"
        )


def _index_docs(corpus, batch_key_id_map, temp_db_path, index_positions, write_lock):
    """
    Index all of the given docs into temp_db_path.

    """

    # pylint: disable=too-many-nested-blocks,too-many-branches

    local_db = db_utilities.connect_sqlite(temp_db_path)

    try:
        # Mapping of {field: {value: (BitMap(), BitMap())}}
        # One bitmap for document occurrence, the other other for recording
        # positional information.
        batch = collections.defaultdict(
            lambda: collections.defaultdict(lambda: (BitMap(), BitMap()))
        )

        # Mapping of fields -> doc_ids, position starts for each document.
        # Note that documents with an empty field present are dropped at this
        # stage.
        field_doc_positions_starts = collections.defaultdict(
            lambda: (BitMap(), BitMap([0]))
        )

        first_doc_id = min(batch_key_id_map.values())
        last_doc_id = max(batch_key_id_map.values())

        for key, doc in corpus.docs(batch_key_id_map):
            doc_id = batch_key_id_map[key]

            doc_features = corpus.doc_to_features(doc)
            for field, values in doc_features.items():
                positional = False

                # handle document value presence for different kinds of fields
                # lists -> positional information to be optionally indexed
                if isinstance(values, list):
                    if index_positions:
                        positional = True

                    # Convert to a set of values for the final document
                    # indexing.
                    doc_values = set(values)

                elif isinstance(values, set):
                    # No work necessary, but we do need to distinguish between
                    # sets and single values so we can convert the latter.
                    doc_values = values
                else:
                    # Convert singleton case to a sequence so the next step
                    # doesn't need to be a special case.
                    doc_values = [values]

                for value in doc_values:
                    if value is None:
                        raise ValueError("Values cannot contain None")
                    batch[field][value][0].add(doc_id)

                # Construct the positional index if needed.
                if positional:
                    batch_position = field_doc_positions_starts[field][1][-1]

                    for position, value in enumerate(values):
                        batch[field][value][1].add(position + batch_position)

                    # If values is empty, move on to the next layer.
                    if not values:
                        continue

                    field_doc_positions_starts[field][0].add(doc_id)
                    # pylint will complain about this as position may not
                    # be defined if values is empty, however if values is
                    # empty we will already have continued.

                    # pylint: disable=undefined-loop-variable
                    field_doc_positions_starts[field][1].add(
                        # +1 because it's the start of the *next* doc.
                        position
                        + batch_position
                        + 1
                    )
                    # pylint: enable=undefined-loop-variable

        with write_lock:
            local_db.execute("pragma synchronous=0")
            local_db.execute("begin")
            local_db.execute(
                """
                CREATE table if not exists inverted_index_segment(
                    field text,
                    value,
                    docs_count,
                    doc_ids roaring_bitmap,
                    position_count,
                    positions roaring_bitmap,
                    first_doc_id
                )
                """
            )

            local_db.execute(
                """
                CREATE table if not exists batch_position(
                    field,
                    first_doc_id,
                    last_doc_id,
                    docs_count,
                    doc_ids roaring_bitmap,
                    doc_position_starts roaring_bitmap,
                    primary key (field, first_doc_id)
                )
                """
            )

            for field, (
                batch_doc_ids,
                position_starts,
            ) in field_doc_positions_starts.items():
                local_db.execute(
                    "insert into batch_position values(?, ?, ?, ?, ?, ?)",
                    (
                        field,
                        first_doc_id,
                        last_doc_id,
                        len(batch_doc_ids),
                        batch_doc_ids,
                        position_starts,
                    ),
                )

            field_order = sorted(batch.keys())

            for field in field_order:
                values = batch[field]

                local_db.executemany(
                    "insert into inverted_index_segment values(?, ?, ?, ?, ?, ?, ?)",
                    (
                        (
                            field,
                            # Use the corpus specified ValueHandler for this field to
                            # transform for the index.
                            corpus.field_values[field].to_index(value),
                            len(docs),
                            docs,
                            len(positions),
                            positions or None,
                            first_doc_id,
                        )
                        for value, (docs, positions) in values.items()
                    ),
                )

            local_db.execute("commit")

    finally:
        local_db.close()

    return temp_db_path


def _calculate_query_cluster_cooccurrence(idx, key, query, cluster_ids):
    with idx:
        weights = []

        for cluster_id in cluster_ids:
            cluster_docs = idx.cluster_docs(cluster_id)
            inter = query.intersection_cardinality(cluster_docs)

            if inter:
                sim = query.jaccard_index(cluster_docs)
                weights.append((cluster_id, sim, inter))

    return key, weights


def _pivot_cluster_features_by_query_jaccard(
    idx, query, cluster_id, top_k, cluster_inter
):
    with idx:
        results = [(0, -1, "", "")] * top_k

        q = len(query)

        features = (
            (min(f[-1], cluster_inter) / (q + f[-1] - min(f[-1], cluster_inter)), *f)
            for f in idx.cluster_features(cluster_id)
        )

        search_order = sorted(features, reverse=True)

        for max_threshold, f_id, field, value, _ in search_order:
            # Early break if the length threshold can't be reached.
            if max_threshold < results[0][0]:
                break

            heapq.heappushpop(
                results, (query.jaccard_index(idx[f_id]), f_id, field, value)
            )

        results = sorted(
            ((*r[1:], r[0]) for r in results if r[0] > 0),
            reverse=True,
            key=lambda r: r[3],
        )

        # Finally compute the similarity of the query with the cluster.
        similarity = query.jaccard_index(idx.cluster_docs(cluster_id))

    return cluster_id, similarity, results


def measure_feature_contribution_to_cluster(
    idx,
    group_key: Any,
    feature_group: set[int],
    add_features: set[int],
    probe_query: Optional[AbstractBitMap],
) -> tuple[
    Any,
    float,
    array.array("q"),
    array.array("d"),
    array.array("q"),
    array.array("d"),
]:
    """
    Measure the contribution of each feature to this cluster.

    The contribution is the delta between the objective of the cluster without
    the feature and with the feature.

    This function also has the side effect of approximating the objective
    contribution for this feature in this cluster (assuming moving only that
    feature).

    """

    with idx:
        # FIRST PHASE: compute the objective and minimal cover stats for the
        # current cluster.

        # The union of all docs covered by the cluster
        cluster_union = BitMap()
        # The set of all docs covered at least twice.
        # This will be used to work out which documents are only covered once.
        covered_twice = BitMap()

        hits = 0
        n_features = len(feature_group)

        if not n_features:
            return (
                group_key,
                0,
                array.array("q"),
                array.array("d"),
                array.array("q"),
                array.array("d"),
            )

        # Construct the union of all cluster tokens, and also the set of
        # documents only covered by a single feature.
        for feature in feature_group:
            docs = idx[feature]

            if probe_query:
                docs &= probe_query

            hits += len(docs)

            # Docs covered at least twice
            covered_twice |= cluster_union & docs
            # All docs now covered
            cluster_union |= docs

        only_once = cluster_union - covered_twice

        c = len(cluster_union)
        objective = hits / (c + n_features) - (hits / (hits + n_features))

        # PHASE 2: compute the incremental change in objective from removing
        # each feature (alone) from the current cluster. Note: using an array
        # to only manage two objects worth of de/serialisation

        remove_feature = array.array("q", feature_group)
        remove_delta = array.array("d", (0 for _ in remove_feature))

        # Features that are already in the cluster, so we need to calculate a remove operator.
        # Effectively we're counting the negative of the score for removing that feature
        # as the effect of adding it to the cluster.
        for i, feature in enumerate(remove_feature):
            docs = idx[feature]

            if probe_query:
                docs &= probe_query

            feature_hits = len(docs)

            old_hits = hits - feature_hits
            only_once_hits = docs.intersection_cardinality(only_once)
            old_c = c - only_once_hits

            # Check if this feature intersects with any other feature in this cluster
            intersects_with_other_feature = only_once_hits < feature_hits

            # It's okay for the cluster to become empty - we'll just prune it.
            if old_c and intersects_with_other_feature:
                old_objective = old_hits / (old_c + (n_features - 1)) - (
                    old_hits / (old_hits + n_features - 1)
                )

                delta = objective - old_objective

            # Penalises features that don't intersect with other features in the cluster.
            elif old_c:
                delta = -1
            # If it would otherwise be a singleton cluster, just mark it as no change
            else:
                delta = 0

            remove_delta[i] = delta

        # PHASE 3: Incremental delta from adding new features to the cluster.
        add_feature = array.array("q", sorted(add_features - feature_group))
        add_delta = array.array("d", (0 for _ in add_feature))

        # All tokens that are adds (not already in the cluster)
        for i, feature in enumerate(add_feature):
            docs = idx[feature]

            if probe_query:
                docs &= probe_query

            feature_hits = len(docs)

            if docs.intersect(cluster_union):
                new_hits = hits + feature_hits
                new_c = docs.union_cardinality(cluster_union)
                new_objective = new_hits / (new_c + (n_features + 1)) - (
                    new_hits / (new_hits + n_features + 1)
                )

                delta = new_objective - objective

            # If the feature doesn't intersect with the cluster at all,
            # give it a bad delta.
            else:
                delta = -1

            add_delta[i] = delta

    return group_key, objective, remove_feature, remove_delta, add_feature, add_delta


def _union_query(args):
    idx, query_key, feature_ids = args

    with idx:
        query = BitMap()
        weight = 0

        for feature_id in feature_ids:
            docs = idx[feature_id]
            query |= docs
            weight += len(docs)

    return query_key, query, weight


def _field_proximity_query(idx, field, clause_feature_ids, first_doc_id, window_size):
    """
    Return documents where values occur near each other in a field.

    `value_clauses` is a list of clauses in DNF: at least one value from each clause
    must be within window_size of each for that position and document to match.

    """

    with idx:

        _, doc_ids, doc_boundaries = idx._get_partition_header(field, first_doc_id)

        positions = idx._union_position_query(first_doc_id, clause_feature_ids[0])
        valid_window = utilities.expand_positions_window(
            positions, doc_boundaries, window_size
        )

        for clause in clause_feature_ids[1:]:

            # Keep only the union positions that actually intersect with the
            # existing window - these are the starts and ends of spans that
            # can satisfy the spacing criteria.
            clause_positions = (
                idx._union_position_query(first_doc_id, clause) & valid_window
            )

            # Early termination if there's no matches at any point during a clause.
            if not clause_positions:
                return BitMap()

            # Keep the remaining positions so we can count them as matching
            positions |= clause_positions

            # Update the windows by intersecting with the new valid windows
            # from this clause.
            valid_window &= utilities.expand_positions_window(
                clause_positions, doc_boundaries, window_size
            )

        # Apply the final valid window to the positions.
        positions &= valid_window

        if not positions:
            return BitMap()

        # Convert matching positions to doc_ids containing the match by looking up
        # in the header for this partition
        matching_docs = BitMap(
            doc_ids[doc_boundaries.rank(position) - 1] for position in positions
        )

        return matching_docs
