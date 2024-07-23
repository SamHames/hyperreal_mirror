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
from collections.abc import Sequence
from functools import wraps
from typing import Any, Hashable, Optional, Union

from pyroaring import AbstractBitMap, BitMap

from hyperreal import _index_schema, db_utilities, utilities
from hyperreal.corpus import Corpus

logger = logging.getLogger(__name__)


class IndexingError(AttributeError):
    "Raised for specific problems during indexing."


FeatureKey = tuple[str, Hashable]
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
                    # TODO: this whole thing can be simplified now that clusters are
                    # immutable.
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
    def __getitem__(self, key: FeatureKey) -> BitMap:
        """
        Retrieve the set of documents matching a literal feature from the index.

        A feature is represented as a tuple of `("field_name", value)`.

        Note that an exception will be raised if the field doesn't exist, but not
        if the value doesn't exist on a valid field. If the value doesn't exist on a
        field the return result indicates no matching documents.

        """

        if isinstance(key, tuple):
            try:
                field, value = key

                # It would be more natural to check in, but there might be cases when
                # fields need to be generated on demand - such as for the EmptyCorpus
                converter = self.field_values[field]
                indexed_value = converter.to_index(value)

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
            raise ValueError("Must provide a ('field', value) pair.")

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

            # Drop existing index tables
            self.db.execute("delete from position_doc_map")
            self.db.execute("delete from position_index")
            self.db.execute("delete from inverted_index")

            # Make sure all of the batches have completed.
            for f in cf.as_completed(futures):
                f.result()

            self.logger.info("Batches complete - merging into main index.")

            # Now merge back to the original index
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
                replace into inverted_index(field, value, docs_count, doc_ids)
                    select
                        field,
                        value,
                        sum(docs_count) as docs_count,
                        roaring_union(doc_ids) as doc_ids
                    from inverted_index_segment iis
                    group by field, value
                    -- Order is an insert optimisation and not strictly necessary
                    order by field, value
                """
            )

            # Position index
            self.db.execute(
                """
                INSERT into position_index
                    (field, value , first_doc_id, position_count, positions)
                    select
                        field,
                        value,
                        first_doc_id,
                        position_count,
                        positions
                    from inverted_index_segment iis
                    where position_count > 0
                    -- Order is an insert optimisation and not strictly necessary
                    order by field, value, first_doc_id
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
                        where (ii.field, ii.value) = (
                            feature_cluster.field, feature_cluster.value
                        )
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

            self._update_cluster_docs(self.cluster_ids)

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
    def restore_deleted_clusters(self, cluster_ids):
        """
        Restore the specified deleted clusters.

        If the cluster ID doesn't exist or it isn't deleted, this operation will do
        nothing.

        """
        self.db.executemany(
            """
            update cluster 
                set deleted_at = null
            where cluster_id = ?
            """,
            [[c] for c in cluster_ids],
        )

        self._update_cluster_docs(cluster_ids)

        self.logger.info(f"Deleted clusters {cluster_ids}.")

    @atomic(writes=True)
    def delete_clusters(self, cluster_ids):
        """
        Delete the specified clusters.

        Deletes are soft deletes only and can be restored using
        restore_deleted_clusters.

        """
        self.db.executemany(
            """
            update cluster set 
                deleted_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                docs_count = 0,
                doc_ids = null
            where cluster_id = ?
            """,
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

    def _update_cluster_docs(self, cluster_ids):
        """
        Update the doc_ids and docs_count for the given clusters.

        This is called automatically by the create_cluster_from_features and rebuild
        methods, and shouldn't need to be called manually. The only reason to call it
        manually is if you have manually update the cluster_feature table, and the
        materialised cluster query is out of date.

        """

        for cluster_id in cluster_ids:
            features = self.cluster_features(cluster_id)

            doc_ids = BitMap()

            for f, _ in features:
                doc_ids |= self[f]

            docs_count = len(doc_ids)

            self.db.execute(
                """
                update cluster set 
                    docs_count = ?,
                    doc_ids = ?
                where cluster_id = ?
                """,
                [docs_count, doc_ids, cluster_id],
            )

            # Also update the docs_counts in the context of the features in the query
            # too. This measures the effective hits of this feature in the context of
            # this query - for boolean OR queries this is still the same as the number
            # of documents matching this query by itself.
            for (field, value), _ in features:
                inter_count = doc_ids.intersection_cardinality(self[(field, value)])
                self.db.execute(
                    """
                    update cluster_feature
                    set docs_count = ?
                    where (cluster_id, field, value) = (?, ?, ?)
                    """,
                    [
                        inter_count,
                        cluster_id,
                        field,
                        self.field_values[field].to_index(value),
                    ],
                )

    @atomic(writes=True)
    def create_cluster_from_features(self, features, name="", notes=""):
        """
        Create a new cluster from the given set of features, returns the new cluster_id.

        Clusters are immutable and apart from the name and notes can only be created or
        soft deleted.

        """

        self.db.execute(
            "insert into cluster(name, notes) values(?, ?)",
            (name, notes),
        )

        cluster_id = list(self.db.execute("select last_insert_rowid()"))[0][0]

        index_features = [
            (
                cluster_id,
                field,
                self.field_values[field].to_index(value),
            )
            for field, value in features
        ]

        self.db.executemany(
            "insert into cluster_feature(cluster_id, field, value) values (?, ?, ?)",
            index_features,
        )

        self._update_cluster_docs([cluster_id])

        self.logger.info(f"Created cluster {cluster_id} from {len(features)} features.")

        return cluster_id

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
                """
                select cluster_id 
                from cluster 
                where deleted_at is null 
                order by cluster_id
                """
            )
        ]

    @atomic()
    def top_cluster_features(self, top_k=20):
        """Return the top_k features according to the number of matching documents."""

        cluster_docs = self.db.execute(
            """
            select cluster_id, docs_count
            from cluster
            where deleted_at is null
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

    def field_features(
        self, field: str, top_k: Optional[int] = None, min_docs_count: int = 1
    ):
        """
        Returns the features in a given field, in descending order of the count of docs.

        If top_k is specified, only the top_k most frequent features by document count
        are returned in descending order.

        """

        top_k = top_k or 2**62

        features = [
            ((row[0], self.field_values[row[0]].from_index(row[1])), row[2])
            for row in self.db.execute(
                """
                select
                    field,
                    value,
                    docs_count
                from inverted_index
                where field = ? and docs_count >= ?
                order by docs_count desc, value
                limit ?
                """,
                [field, min_docs_count, top_k],
            )
        ]

        return features

    def cluster_annotations(self, cluster_id):

        try:
            return list(
                self.db.execute(
                    "select name, notes from cluster where cluster_id = ?", [cluster_id]
                )
            )[0]
        except IndexError:
            raise KeyError(f"Cluster with id {cluster_id=} does not exist.")

    def update_cluster_annotations(self, cluster_id, name=None, notes=None):
        """
        Update the name and notes for a given cluster.

        name and notes are both optional: if left as the default None, the name or notes
        field will be left unchanged. To empty the field, use the empty string
        explicitly.

        Currently silently continues if cluster_id does not exist.

        TODO: this might need to check for deletion as well?

        """
        self.db.execute(
            """
            update cluster set
                -- If name or notes are null, use the existing field values.
                name = coalesce(?, name),
                notes = coalesce(?, notes)
            where cluster_id = ?
            """,
            [name, notes, cluster_id],
        )

    def cluster_features(self, cluster_id, top_k=2**62):
        """
        Returns an impact ordered list of features for the given cluster.

        If top_k is specified, only the top_k most frequent features by
        document count are returned in descending order.

        """
        cluster_features = [
            ((row[0], self.field_values[row[0]].from_index(row[1])), row[2])
            for row in self.db.execute(
                """
                select
                    field,
                    value,
                    -- Note that docs_count is denormalised to allow
                    -- a per cluster sorting of document count.
                    docs_count
                from cluster_feature
                where cluster_id = ?
                order by docs_count desc
                limit ?
                """,
                [cluster_id, top_k],
            )
        ]

        return cluster_features

    @atomic()
    def union_bitslice(self, features: Sequence[FeatureKey]):
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

        features = [r[0] for r in self.cluster_features(cluster_id)]

        return self.union_bitslice(features)

    def cluster_docs(self, cluster_id: int) -> AbstractBitMap:
        """Return the bitmap of documents covered by this cluster."""
        return list(
            self.db.execute(
                "select doc_ids from cluster where cluster_id = ?", [cluster_id]
            )
        )[0][0]

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

        futures = set()

        for first_doc_id in self._field_partitions(field):
            futures.add(
                self.pool.submit(
                    _field_proximity_query,
                    self,
                    field,
                    value_clauses,
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
    def _union_position_query(self, field, first_doc_id, values):
        positions = BitMap()

        for value in values:
            positions |= self._get_partition_positions(field, first_doc_id, value)

        return positions

    def _get_partition_positions(self, field, first_doc_id, value):
        positions = list(
            self.db.execute(
                """
                select
                    positions
                from position_index
                where (field, value, first_doc_id) = (?, ?, ?)
                """,
                [field, self.field_values[field].to_index(value), first_doc_id],
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


class _SortableFeatureSimilarity(tuple):
    """
    A slight customisation of tuple to sort only on the numerical values not features.

    This means the sort isn't stable, but also means we don't have to worry about
    comparing incomparable values across different fields to break ties.

    This is necessary because you can't pass a custom comparison key function to
    heapq.heappushpop.

    """

    def __lt__(self, other):
        return self[0] < other[0]


def _pivot_cluster_features_by_query_jaccard(
    idx, query, cluster_id, top_k, cluster_inter
):

    with idx:
        results = [((-1, -math.inf), (-1, -1))] * top_k

        q = len(query)

        features = (
            _SortableFeatureSimilarity(
                (
                    min(f[-1], cluster_inter) / (q + f[-1] - min(f[-1], cluster_inter)),
                    *f,
                )
            )
            for f in idx.cluster_features(cluster_id)
        )

        search_order = sorted(features, reverse=True)

        for max_threshold, feature, _ in search_order:
            # Early break if the length threshold can't be reached.
            if max_threshold < results[0][0][0]:
                break

            feature_docs = idx[feature]

            heapq.heappushpop(
                results,
                # We'll break ties in the direction of smaller intersections.
                # This still isn't a perfectly stable sort though.
                _SortableFeatureSimilarity(
                    (
                        (
                            query.jaccard_index(feature_docs),
                            -query.intersection_cardinality(feature_docs),
                        ),
                        feature,
                    )
                ),
            )

        # Not completely deterministic ordering: we don't want to sort by arbitrary
        # and potentially unsortable features, so just use the similarity score alone.
        results = [(r[1], r[0][0]) for r in sorted(results, reverse=True)]

        # Finally compute the similarity of the query with the cluster.
        similarity = query.jaccard_index(idx.cluster_docs(cluster_id))

    return cluster_id, similarity, results


def _union_query(args):
    idx, query_key, features = args

    with idx:
        query = BitMap()
        weight = 0

        for feature in features:
            docs = idx[feature]
            query |= docs
            weight += len(docs)

    return query_key, query, weight


def _field_proximity_query(idx, field, value_clauses, first_doc_id, window_size):
    """
    Return documents where values occur near each other in a field.

    `value_clauses` is a list of clauses in DNF: at least one value from each clause
    must be within window_size of each for that position and document to match.

    """

    with idx:

        _, doc_ids, doc_boundaries = idx._get_partition_header(field, first_doc_id)

        positions = idx._union_position_query(field, first_doc_id, value_clauses[0])
        valid_window = utilities.expand_positions_window(
            positions, doc_boundaries, window_size
        )

        for clause in value_clauses[1:]:

            # Keep only the union positions that actually intersect with the
            # existing window - these are the starts and ends of spans that
            # can satisfy the spacing criteria.
            clause_positions = (
                idx._union_position_query(field, first_doc_id, clause) & valid_window
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
