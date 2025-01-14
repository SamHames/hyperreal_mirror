"""

indexable_docs format

rebuild is the core interface -- and is grafted onto the main Index object. Everything
else comes from the provided corpus.

Citation: the flat index position format.
Citation: roaring bitmaps.

Pointer to a description of the index layout on disk - possibly in the docs folder?

"""

import collections
import concurrent.futures as cf
import contextlib
import dataclasses as dc
import math
import os
import tempfile
from typing import Iterable, Optional

from pyroaring import BitMap, BitMap64

from . import corpus, db_utilities

__all__ = ["build_index"]


def build_index(
    db_path,
    corpus: corpus.Corpus,
    pool: cf.Executor,
    doc_batch_size: int = 1000,
    max_workers: Optional[int] = None,
):
    """
    (Re)Build the index of the given corpus at the given database.

    max_workers will aim to use all available workers in the pool by default.

    """

    try:
        # Try to infer the number of workers from the provided pool, but since it's not
        # public, handle where it might be missing.
        max_workers = pool._max_workers or max_workers
    except AttributeError:
        # TODO: log the fallthrough
        max_workers = max_workers or 4

    with tempfile.TemporaryDirectory() as tempdir:

        # Initialise the index to merge everything else into
        merge_segments = os.path.join(tempdir, "temp_index.db")
        temp_db = _init_segment(merge_segments)
        temp_db.close()

        # First dispatch contiguous segments for indexing, one whole range per worker.
        total_docs = len(corpus)
        worker_batch_size = math.ceil(total_docs / max_workers)

        current_doc_id = 0
        segment_in_progress = set()

        # We'll generate one segment per batch/worker - then we'll dispatch these to
        # the pool for indexing in parallel.
        for doc_key_batch in _batch(corpus.all_doc_keys(), worker_batch_size):

            segment_path = os.path.join(tempdir, str(current_doc_id))

            segment_in_progress.add(
                pool.submit(
                    _make_segment,
                    corpus,
                    segment_path,
                    current_doc_id,
                    doc_key_batch,
                    doc_batch_size,
                )
            )

            # dispatch batch and current_doc_id for indexing
            current_doc_id += len(doc_key_batch)

        for segment in cf.as_completed(segment_in_progress):
            start_doc_id, merge_from = segment.result()
            _merge_into(start_doc_id, merge_from, merge_segments)

        _finalise_into(merge_segments, db_path)


def _batch(sequence, batch_size):
    """A generator for batches up to the given size."""

    current_batch = []
    current_size = 0

    for item in sequence:
        current_batch.append(item)
        current_size += 1

        if current_size == batch_size:
            yield current_batch
            current_batch = []
            current_size = 0

    if current_size:
        yield current_batch


def _init_segment(db_path):
    """Initialise the database with the minimum set of tables for a segment."""
    db = db_utilities.connect_sqlite(db_path)

    db.executescript(
        """
        CREATE table if not exists doc_key (
            doc_id integer primary key,
            doc_key unique
        );

        CREATE table if not exists inverted_index_segment(
            field text,
            value not null,
            first_doc_id,
            doc_count,
            doc_ids roaring_bitmap,
            position_count,
            positions roaring_bitmap64,
            primary key (field, value, first_doc_id)
        );

        CREATE table if not exists segment_header(
            field,
            first_doc_id,
            max_cardinality,
            value_handler_name,
            stored_sorted,
            doc_count,
            doc_ids roaring_bitmap,
            position_count,
            position_doc_ids roaring_bitmap,
            position_doc_starts roaring_bitmap64,
            primary key (field, first_doc_id)
        );

        """
    )

    return db


def _make_segment(
    corpus: corpus.Corpus,
    segment_path,
    start_doc_id: int,
    doc_keys: Iterable[corpus.DocKey],
    doc_batch_size: int,
):
    """Make an on-disk segment representing the index for this group of documents."""

    db = _init_segment(segment_path)

    # indicator for where the field is up to in terms of position count. All docs
    # in this segment will have non-overlapping position ranges.
    field_positions = collections.Counter()

    # Observed handlers for each field - in conjunction with the cardinalities this
    # will be used both as the descriptive schema of what was indexed, but also
    # used to validate that there is only one ValueHandler per field.
    observed_field_handlers = collections.defaultdict(set)

    next_doc_id = start_doc_id

    db.execute("begin")

    for batch_docs in _batch(doc_keys, doc_batch_size):

        # Prepare the batch of document for insertion, generating an in-memory version
        # of the index.
        (
            batch_keys,
            batch_segment,
            batch_segment_header,
            insert_order,
            field_positions,
        ) = _prepare_doc_batch(corpus, next_doc_id, field_positions, batch_docs)

        # Write this batch to the staging segment.
        _stage_doc_batch(
            db,
            next_doc_id,
            batch_keys,
            batch_segment,
            batch_segment_header,
            insert_order,
        )

        # We also need to check this at the end - but it's better to fail as early as
        # possible if anything is wrong.
        invalid_fields = [
            (field, handlers)
            for field, handlers in observed_field_handlers.items()
            if len(handlers) > 1
        ]

        if invalid_fields:
            raise SchemaValidationError(
                "Only one ValueHandler can be used per field across all documents.\n"
                "The following fields are invalid for indexing:\n"
                "\n".join(
                    f"\t{field=} has {handlers=}" for field, handlers in invalid_fields
                )
            )

        next_doc_id += len(batch_docs)

    db.execute("commit")

    return start_doc_id, segment_path


@dc.dataclass
class _DocsPositions:
    docs: BitMap = dc.field(default_factory=BitMap)
    positions: BitMap64 = dc.field(default_factory=BitMap64)


@dc.dataclass
class _FieldBatchHeader:
    """Records all the information about this field in the batch."""

    docs: BitMap = dc.field(default_factory=BitMap)
    docs_w_positions: BitMap = dc.field(default_factory=BitMap)
    doc_position_ranges: BitMap64 = dc.field(default_factory=BitMap64)
    max_cardinality: int = 0
    value_handler_name: str = ""
    stored_sorted: bool = False


def _prepare_doc_batch(corpus, start_doc_id, field_positions, doc_keys):
    """
    Process this group of docs into a Python structure suitable for serialising.

    This includes gathering and evaluating the schema as we go.

    """

    # Mapping of {field: {value: (BitMap(), BitMap64())}}
    # One bitmap for document occurrence, the other other for recording
    # positional information.
    batch_segment = collections.defaultdict(
        lambda: collections.defaultdict(_DocsPositions)
    )
    # Mapping of fields -> doc_ids, doc_ids w position starts, position starts for
    # each document. Note this requires recording three pieces of information:
    # Whether the field was present on the doc, regardless of content, and whether
    # the doc has any positional information. The latter two will be empty for most
    # fields that are simple attributes or sets.
    batch_segment_header = collections.defaultdict(_FieldBatchHeader)

    doc_id = start_doc_id

    batch_keys = []

    for doc_key, indexable_doc in corpus.indexable_docs(doc_keys):

        # We could just pass through doc_keys, but there might be cases where it
        # makes sense for say indexable_docs to ignore invalid keys and continue.
        batch_keys.append(doc_key)

        for field, values in indexable_doc.items():

            batch_field = batch_segment[field]

            if isinstance(values, list):
                # Prepare the values for indexing at the doc level
                doc_values = set(values)
                position_count = value_count = len(values)
                container_type = "list"

            elif isinstance(values, set):
                doc_values = values
                value_count = len(values)
                position_count = 0
                container_type = "set"

            else:
                doc_values = [values]
                value_count = 1
                position_count = 0
                container_type = "none"

            # Accumulate the cardinalities observed (as-indexed).
            batch_segment_header[field].max_cardinality = max(
                batch_segment_header[field].max_cardinality, value_count
            )

            # Actually index the values at the document level.
            for value in doc_values:
                batch_field[value].docs.add(doc_id)

            # Record the presence of this field in the document, regardless of
            # whatever funky thing is happening with indexing.
            batch_segment_header
            batch_segment_header[field].docs.add(doc_id)

            # If there are no positions to record, we don't record anything for this
            # doc here. This means there are some docs that will appear in the first
            # part of the header, because the field was present, but will not appear
            # here because there were no positions to record.
            if position_count:

                next_position = start_position = field_positions[field]

                for value in values:
                    batch_field[value].positions.add(next_position)
                    next_position += 1

                # Keep track of the doc-positional mapping
                batch_segment_header[field].docs_w_positions.add(doc_id)
                # Annotate the position ranges for this field in the doc. Always add
                # the [start, end) range - this will be redundant for all of the
                # instances except the first and last.
                batch_segment_header[field].doc_position_ranges.add(start_position)
                batch_segment_header[field].doc_position_ranges.add(next_position)
                field_positions[field] = next_position

        doc_id += 1

    # Create a mapping for all of the types from Python to SQLite representations, along
    # with the insert order. Together with the cardinalities and containers this allows
    # us to generate the description of the indexed fields in this document.
    # We'll also map the different types to value handlers so we can load the schema
    # later.

    field_order = sorted(batch_segment.keys())
    insert_order = dict()

    for field in field_order:

        field_values = batch_segment[field].keys()

        seen_handlers = set()

        # Establish the order of values, validating the constraint that there is
        # only one ValueHandler per field.
        value_types = {type(v) for v in field_values}
        handlers = {corpus.type_handlers[v] for v in value_types}

        if len(handlers) > 1:
            # TODO: Write this error message more proper.
            raise corpus.SchemaValidationError(
                "TODO - a field is constrained to only be handled by one value handler"
            )

        handler = handlers.pop()
        transform = handler.to_index

        batch_segment_header[field].value_handler_name = handler.value_name
        batch_segment_header[field].stored_sorted = handler.stored_sorted

        value_order = sorted((transform(value), value) for value in field_values)

        if any(v[0] is None for v in value_order):
            raise ValueError("Field values cannot be None when writing to the DB.")

        insert_order[field] = value_order

    # TODO construct the schema representation in memory
    return (
        batch_keys,
        batch_segment,
        batch_segment_header,
        insert_order,
        field_positions,
    )


def _stage_doc_batch(
    db, first_doc_id, batch_keys, batch_segment, batch_segment_header, insert_order
):
    """
    Stage the given doc batch to the on disk segment.

    """

    field_order = sorted(batch_segment.keys())

    # First stage the keys
    db.executemany(
        "insert into doc_key values(?, ?)",
        zip(range(first_doc_id, first_doc_id + len(batch_keys)), batch_keys),
    )

    for field, value_order in insert_order.items():
        field_batch = batch_segment[field]

        db.executemany(
            "INSERT into inverted_index_segment values(?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    field,
                    index_value,
                    first_doc_id,
                    len(field_batch[value].docs),
                    field_batch[value].docs,
                    len(field_batch[value].positions),
                    field_batch[value].positions,
                )
                for index_value, value in value_order
            ),
        )

        # Don't forget to write the header row describing the position-doc mapping
        header_row = batch_segment_header[field]

        n_positions = 0
        if header_row.doc_position_ranges:
            n_positions = header_row.doc_position_ranges[-1]

        db.execute(
            "INSERT into segment_header values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                field,
                first_doc_id,
                header_row.max_cardinality,
                header_row.value_handler_name,
                header_row.stored_sorted,
                len(header_row.docs),
                header_row.docs,
                n_positions,
                header_row.docs_w_positions,
                header_row.doc_position_ranges,
            ),
        )


def _merge_into(first_doc_id, from_segment, to_segment):
    """
    Merge the contents of this index into a single segment on `other_index`.

    This assumes that all document + position ranges for this segment are
    contiguous.

    Note that this method makes no effort to worry about concurrent access from
    other processes when merging - you'll either need to call this from the main
    thread, or use some other mechanism to manage contention.

    """

    db = _init_segment(from_segment)

    with contextlib.closing(db):
        try:
            db.execute("attach ? as 'merge_segment'", [str(to_segment)])
            db.execute("begin")

            # Merge the doc keys in
            db.execute(
                """
                INSERT into merge_segment.doc_key select * from doc_key
                """
            )

            # Then merge both the inverted index segments and the associated header.
            # Because the positions don't overlap, this can all be done with a
            # simple union of the underlying bitmaps.
            db.execute(
                """
                INSERT into merge_segment.inverted_index_segment
                    select
                        field,
                        value,
                        ?,
                        sum(doc_count),
                        roaring_union(doc_ids),
                        sum(position_count),
                        roaring_union64(positions)
                    from inverted_index_segment
                    group by field, value
                    """,
                [first_doc_id],
            )

            db.execute(
                """
                INSERT into merge_segment.segment_header
                    select
                        field,
                        ?,
                        max(max_cardinality),
                        value_handler_name,
                        stored_sorted,
                        sum(doc_count),
                        roaring_union(doc_ids),
                        sum(position_count),
                        roaring_union(position_doc_ids),
                        roaring_union64(position_doc_starts)
                    from segment_header
                    -- Note that group by value_handler_name is redundant if this is well
                    -- formed, but will generate a primary key error if the invariant
                    -- of only one value_handler_name per field is violated.
                    -- TODO: might be a better way to do this?
                    -- TODO: check performance is acceptable for large indexes, it might
                    -- be preferable to not have a primary key on this table and create
                    -- an index after.
                    group by field, value_handler_name
                    """,
                [first_doc_id],
            )

            db.execute("commit")

        except Exception as e:
            db.execute("rollback")
            raise


def _finalise_into(from_segment, to_final):
    """
    Merge everything into the final destination index database.

    This step needs to do some specific additional processing in order to:

    - shifting all the positional information from non-contiguous sections
    - processing field values that are amenable to range encoding

    """

    db = _init_segment(from_segment)

    with contextlib.closing(db):
        try:
            db.execute("attach ? as 'final'", [str(to_final)])
            db.execute("begin")

            # new doc_keys
            db.execute("DELETE from final.doc_key")
            db.execute("INSERT into final.doc_key select * from main.doc_key")

            # Work out how much to shift each positional set for each field/first_doc_id
            # combination.
            db.execute(
                """
                create temporary table field_shift(
                    field,
                    first_doc_id,
                    cumulative_shift,
                    primary key (field, first_doc_id)
                )
                """
            )
            for (field,) in db.execute("select distinct field from segment_header"):

                current_shift = 0
                segment_sizes = db.execute(
                    """
                    SELECT first_doc_id, position_count 
                    from segment_header
                    where field = ?
                    order by first_doc_id
                    """,
                    [field],
                )

                for first_doc_id, position_count in segment_sizes:
                    db.execute(
                        "insert into field_shift values(?, ?, ?)",
                        [field, first_doc_id, current_shift],
                    )
                    current_shift += position_count

            # First process the field summary, so that we can work out the schema
            db.execute("DELETE from final.field_summary")
            db.execute(
                """
                INSERT into final.field_summary 
                    select
                        field, 
                        max(max_cardinality),
                        value_handler_name,
                        -- This is guaranteed to be the same across the field so we can
                        -- rely on SQLite behaviour to grab an arbitrary row.
                        stored_sorted,
                        sum(doc_count),
                        roaring_union(doc_ids),
                        sum(position_count),
                        roaring_union(position_doc_ids),
                        roaring_shift_union64(position_doc_starts, cumulative_shift)
                    from segment_header
                    inner join field_shift using(field, first_doc_id)
                    group by field, value_handler_name
                """
            )

            ## The new inverted index ##

            # Work out which fields are just passed through, and which are range
            # encoded. For range encoded fields we will do an extra processing step -
            # all others can be inserted normally.

            db.execute("DELETE from final.inverted_index")
            field_processing = db.execute(
                """
                SELECT field, max_cardinality, stored_sorted, position_count
                from field_summary
                order by field
                """
            )

            for (
                field,
                max_cardinality,
                stored_sorted,
                position_count,
            ) in field_processing:

                if (max_cardinality, stored_sorted, position_count) == (1, 1, 0):

                    # Generate the initial merged values
                    merged = db.execute(
                        """
                        SELECT
                            field, 
                            value,
                            sum(doc_count),
                            roaring_union(doc_ids),
                            sum(position_count),
                            roaring_shift_union64(positions, cumulative_shift)
                        from inverted_index_segment
                        inner join field_shift using(field, first_doc_id)
                        where field = ?
                        group by value
                        order by value
                        """,
                        [field],
                    )
                    # range encode - a value now represents itself + doc_ids matching
                    # all the values that were lower than this.
                    accumulated_docs = BitMap()

                    for field, value, _, docs, pos_count, positions in merged:

                        accumulated_docs |= BitMap.deserialize(docs)

                        db.execute(
                            "INSERT into final.inverted_index values "
                            "(?, ?, ?, ?, ?, ?)",
                            (
                                field,
                                value,
                                len(accumulated_docs),
                                accumulated_docs,
                                position_count,
                                positions,
                            ),
                        )

                else:
                    # Pass through case - no extra details required.
                    db.execute(
                        """
                        INSERT into final.inverted_index 
                            select
                                field, 
                                value,
                                sum(doc_count),
                                roaring_union(doc_ids),
                                sum(position_count),
                                roaring_shift_union64(positions, cumulative_shift)
                            from inverted_index_segment
                            inner join field_shift using(field, first_doc_id)
                            where field = ?
                            group by value
                        """,
                        [field],
                    )

            db.execute("commit")

        except Exception as e:
            db.execute("rollback")
            raise
