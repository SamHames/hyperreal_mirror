"""

rebuild is the core interface -- and is grafted onto the main Index object. Everything
else comes from the provided corpus.

Citation: the flat index position format.
Citation: roaring bitmaps.

TODO: document indexable_docs format

TODO: Pointer to a description of the index layout on disk - possibly in the docs
folder?

"""

import collections
import concurrent.futures as cf
import contextlib
import dataclasses as dc
import math
import numbers
import os
import tempfile
from typing import Iterable, Optional

from pyroaring import BitMap

from . import corpus, db_utilities, field_types

__all__ = ["build_index"]


def build_index(
    db_path,
    corpus: corpus.HyperrealCorpus,
    pool: cf.Executor,
    doc_batch_size: int = 1000,
    max_workers: Optional[int] = None,
    passage_size=16,
):
    """
    (Re)Build the index of the given corpus at the given database.

    max_workers will aim to use all available workers in the pool by default.

    """

    try:
        # Try to infer the number of workers from the provided pool, but since it's not
        # public, handle where it might be missing.
        max_workers = max_workers or pool._max_workers
    except AttributeError:
        # TODO: log the fallthrough
        max_workers = 4

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
                    passage_size,
                )
            )

            # dispatch batch and current_doc_id for indexing
            current_doc_id += len(doc_key_batch)

        for segment in cf.as_completed(segment_in_progress):
            start_doc_id, merge_from = segment.result()
            _merge_into(start_doc_id, merge_from, merge_segments)

        _finalise_into(merge_segments, db_path, passage_size)


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
            location_count,
            primary key (field, value, first_doc_id)
        );

        CREATE table if not exists inverted_index_segment_score(
            field text references field_summary,
            value not null,
            score integer not null,
            first_doc_id integer not null,
            doc_count,
            doc_ids roaring_bitmap,
            primary key (field, value, score, first_doc_id)
        );

        CREATE table if not exists location_index_segment(
            field text references field_summary,
            value not null,
            mod_location integer not null,
            first_doc_id,
            location_count,
            passage_ids roaring_bitmap,
            primary key (field, value, mod_location, first_doc_id)
        );

        CREATE table if not exists segment_header(
            field,
            first_doc_id,
            value_handler_name,
            max_value_count,
            max_scored_value_count,
            max_location_count,
            range_encodable,
            doc_count,
            doc_ids roaring_bitmap,
            total_location_count,
            passage_count,
            passage_doc_ids roaring_bitmap,
            doc_passage_starts roaring_bitmap,
            primary key (field, first_doc_id)
        );
        """
    )

    return db


def _make_segment(
    corpus: corpus.HyperrealCorpus,
    segment_path,
    start_doc_id: int,
    doc_keys: Iterable[corpus.DocKey],
    doc_batch_size: int,
    passage_size: int,
):
    """Make an on-disk segment representing the index for this group of documents."""

    db = _init_segment(segment_path)

    # indicator for where the field is up to in terms of location count. All docs
    # in this segment will have non-overlapping location ranges.
    field_locations = collections.Counter()

    # Observed handlers for each field - in conjunction with the cardinalities this
    # will be used both as the descriptive schema of what was indexed, but also
    # used to validate that there is only one ValueHandler per field.
    observed_field_handlers = collections.defaultdict(set)

    next_doc_id = start_doc_id

    with contextlib.closing(corpus) as corp:

        db.execute("begin")

        for batch_docs in _batch(doc_keys, doc_batch_size):

            # Prepare the batch of document for insertion, generating an in-memory
            # version of the index.
            (
                batch_keys,
                batch_segment,
                batch_segment_header,
                insert_order,
                field_locations,
            ) = _prepare_doc_batch(
                corp, next_doc_id, field_locations, batch_docs, passage_size
            )

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
                raise corpus.SchemaValidationError(
                    "Only one ValueHandler can be used per field across all documents.\n"
                    "The following fields are invalid for indexing:\n"
                    "\n".join(
                        f"\t{field=} has {handlers=}"
                        for field, handlers in invalid_fields
                    )
                )

            next_doc_id += len(batch_docs)

        db.execute("commit")

    return start_doc_id, segment_path


@dc.dataclass
class _DocsPassages:
    score_docs: dict[int, BitMap] = dc.field(
        default_factory=lambda: collections.defaultdict(BitMap)
    )
    passages: dict[int, BitMap] = dc.field(
        default_factory=lambda: collections.defaultdict(BitMap)
    )


@dc.dataclass
class _FieldBatchHeader:
    """Records all the information about this field in the batch."""

    docs: BitMap = dc.field(default_factory=BitMap)
    docs_w_passages: BitMap = dc.field(default_factory=BitMap)
    doc_passage_ranges: BitMap = dc.field(default_factory=BitMap)
    total_location_count: int = 0
    max_value_count: int = 0
    max_scored_value_count: int = 0
    max_location_count: int = 0
    value_handler_name: str = ""
    range_encodable = True


def _prepare_doc_batch(corpus, start_doc_id, field_locations, doc_keys, passage_size):
    """
    Process this group of docs into a Python structure suitable for serialising.

    This includes gathering and evaluating the schema as we go.

    """

    # One bitmap for document occurrence, the other other for recording
    # location information.
    batch_segment = collections.defaultdict(
        lambda: collections.defaultdict(_DocsPassages)
    )
    # Mapping of fields -> doc_ids, doc_ids w location starts, location starts for
    # each document. Note this requires recording three pieces of information:
    # Whether the field was present on the doc, regardless of content, and whether
    # the doc has any location information. The latter two will be empty for most
    # fields that are simple attributes or sets.
    batch_segment_header = collections.defaultdict(_FieldBatchHeader)

    doc_id = start_doc_id

    batch_keys = []

    for doc_key, doc in corpus.docs(doc_keys):

        doc_features = corpus.doc_to_features(doc)

        batch_keys.append(doc_key)

        for field, values in doc_features.items():

            batch_field = batch_segment[field]
            batch_header = batch_segment_header[field]

            vals = values.values
            scores = values.value_scores
            locs = values.value_locations

            value_count = len(vals)
            scored_value_count = len(scores)
            loc_count = len(locs)

            ## Record properties of the current field
            stored_locations = stored_scores = range_encodable = False

            if isinstance(values, field_types.RangeEncodableValue):
                range_encodable = True

            if scores:
                stored_scores = True

            if locs:
                stored_locations = True

            ## Now index the various information for this field

            # (required) values
            for value in vals:
                batch_field[value].score_docs[None].add(doc_id)

            # (optional) scores
            for value, score in scores:
                # Score must be a real number...
                if isinstance(score, numbers.Real):
                    batch_field[value].score_docs[score].add(doc_id)
                else:
                    raise ValueError(f"Score {score} needs to be a numeric type")

            # (optional locations)
            # Starting offset for the next location in this field
            start_location = field_locations[field]
            next_location = None

            # Index locations if present into passages of fixed size and offset within
            # those passages.
            for value, location in locs:
                next_location = start_location + location
                passage_id, offset = divmod(next_location, passage_size)
                batch_field[value].passages[offset].add(passage_id)

            if next_location is not None:

                # After a field in a document finishes, move the position counter so as
                # to consume the remainder of the current group, and consume the next
                # group, leaving a blank group between. The blank group simplifies all
                # handling of edge cases for both passage and positional queries, as we
                # don't need to worry about overlapping documents unless the checked
                # offset is larger than one passage.

                group, offset = divmod(next_location, passage_size)
                remaining_in_group = passage_size - offset
                next_location += passage_size + remaining_in_group

                assert next_location % passage_size == 0

                # Keep track of the doc-positional mapping
                batch_header.docs_w_passages.add(doc_id)
                # Annotate the group ranges for this field in the doc. Always add
                # the [start, end) range - this will be redundant for all of the
                # instances except the first and last.
                batch_header.doc_passage_ranges.add(start_location // passage_size)
                batch_header.doc_passage_ranges.add(next_location // passage_size)

                batch_header.total_location_count += loc_count

                field_locations[field] = next_location

            # Accumulate header information about what has been indexed, such as the
            # cardinalities observed.
            batch_header.max_value_count = max(
                batch_header.max_value_count, value_count
            )
            batch_header.max_location_count = max(
                batch_header.max_location_count, loc_count
            )

            batch_header.max_scored_value_count = max(
                batch_header.max_scored_value_count, scored_value_count
            )

            batch_header.range_encodable = (
                batch_header.range_encodable & range_encodable
            )

            # Record the presence of this field in the document, regardless of
            # whatever funky thing is happening with indexing.
            batch_header.docs.add(doc_id)

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
        field_locations,
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
            "INSERT into inverted_index_segment values(?, ?, ?, ?, ?, ?)",
            (
                (
                    field,
                    index_value,
                    first_doc_id,
                    len(field_batch[value].score_docs[None]),
                    field_batch[value].score_docs[None],
                    sum(
                        len(passages)
                        for passages in field_batch[value].passages.values()
                    ),
                )
                for index_value, value in value_order
            ),
        )

        db.executemany(
            "INSERT into inverted_index_segment_score values(?, ?, ?, ?, ?, ?)",
            (
                (field, index_value, score, first_doc_id, len(docs), docs)
                for index_value, value in value_order
                for score, docs in field_batch[value].score_docs.items()
                if score is not None
            ),
        )

        db.executemany(
            "INSERT into location_index_segment values(?, ?, ?, ?, ?, ?)",
            (
                (
                    field,
                    index_value,
                    mod_location,
                    first_doc_id,
                    len(passages),
                    passages,
                )
                for index_value, value in value_order
                for mod_location, passages in field_batch[value].passages.items()
            ),
        )

        # Don't forget to write the header row describing the position-doc mapping
        header_row = batch_segment_header[field]

        passage_count = 0

        if header_row.doc_passage_ranges:
            passge_count = header_row.doc_passage_ranges[-1]

        db.execute(
            "INSERT into segment_header values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                field,
                first_doc_id,
                header_row.value_handler_name,
                header_row.max_value_count,
                header_row.max_scored_value_count,
                header_row.max_location_count,
                header_row.range_encodable,
                len(header_row.docs),
                header_row.docs,
                header_row.total_location_count,
                passage_count,
                header_row.docs_w_passages,
                header_row.doc_passage_ranges,
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
                        sum(location_count)
                    from inverted_index_segment
                    group by field, value
                    """,
                [first_doc_id],
            )

            db.execute(
                """
                INSERT into merge_segment.inverted_index_segment_score
                    select
                        field,
                        value,
                        score,
                        ?,
                        sum(doc_count),
                        roaring_union(doc_ids)
                    from inverted_index_segment_score
                    group by field, value, score
                    """,
                [first_doc_id],
            )

            db.execute(
                """
                INSERT into merge_segment.location_index_segment
                    select
                        field,
                        value,
                        mod_location,
                        ?,
                        sum(location_count),
                        roaring_union(passage_ids)
                    from location_index_segment
                    group by field, value, mod_location
                    """,
                [first_doc_id],
            )

            db.execute(
                """
                INSERT into merge_segment.segment_header
                    select
                        field,
                        ?,
                        value_handler_name,
                        max(max_value_count),
                        max(max_scored_value_count),
                        max(max_location_count),
                        min(range_encodable),
                        sum(doc_count),
                        roaring_union(doc_ids),
                        sum(total_location_count),
                        max(passage_count),
                        roaring_union(passage_doc_ids),
                        roaring_union(doc_passage_starts)
                    from segment_header
                    -- Note that group by value_handler_name is redundant if this is well
                    -- formed, but will generate a primary key error if the invariant
                    -- of only one value_handler_name per field is violated.
                    group by field, value_handler_name
                    """,
                [first_doc_id],
            )

            db.execute("commit")

        except Exception as e:
            db.execute("rollback")
            raise


def _finalise_into(from_segment, to_final, passage_size):
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

            db.execute(
                "replace into index_setting values(?, ?)",
                ("passage_size", passage_size),
            )

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
                    SELECT first_doc_id, passage_count 
                    from segment_header
                    where field = ?
                    order by first_doc_id
                    """,
                    [field],
                )

                for first_doc_id, passage_count in segment_sizes:
                    db.execute(
                        "insert into field_shift values(?, ?, ?)",
                        [field, first_doc_id, current_shift],
                    )
                    current_shift += passage_count

            # First process the field summary, so that we can work out the schema
            db.execute("DELETE from final.field_summary")
            db.execute(
                """
                INSERT into final.field_summary 
                    select
                        field, 
                        value_handler_name,
                        -- These are value summary rows - we'll update them after 
                        -- the next step.
                        null,
                        null,
                        null,
                        min(range_encodable),
                        max(max_value_count),
                        max(max_scored_value_count),
                        max(max_location_count),
                        sum(total_location_count),
                        sum(doc_count),
                        roaring_union(doc_ids),
                        sum(passage_count),
                        roaring_union(passage_doc_ids),
                        roaring_shift_union(doc_passage_starts, cumulative_shift)
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
            db.execute("DELETE from final.inverted_index_score")
            db.execute("DELETE from final.location_index")
            field_processing = db.execute(
                """
                SELECT field, max_value_count, range_encodable
                from field_summary
                order by field
                """
            )

            for (
                field,
                max_value_count,
                range_encodable,
            ) in field_processing:

                if (max_value_count, range_encodable) == (1, 1):

                    # Generate the initial merged values
                    merged = db.execute(
                        """
                        SELECT
                            field, 
                            value,
                            sum(doc_count),
                            roaring_union(doc_ids),
                            sum(location_count)
                        from inverted_index_segment
                        where field = ?
                        group by value
                        order by value
                        """,
                        [field],
                    )
                    # range encode - a value now represents itself + doc_ids matching
                    # all the values that were lower than this.
                    accumulated_docs = BitMap()

                    for field, value, _, docs, location_count in merged:

                        accumulated_docs |= BitMap.deserialize(docs)

                        db.execute(
                            "INSERT into final.inverted_index values "
                            "(?, ?, ?, ?, ?)",
                            (
                                field,
                                value,
                                len(accumulated_docs),
                                accumulated_docs,
                                location_count,
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
                                sum(location_count)
                            from inverted_index_segment
                            inner join field_shift using(field, first_doc_id)
                            where field = ?
                            group by value
                            order by value
                        """,
                        [field],
                    )

                    db.execute(
                        """
                        INSERT into final.inverted_index_score 
                            select
                                field, 
                                value,
                                score,
                                sum(doc_count),
                                roaring_union(doc_ids)
                            from inverted_index_segment_score
                            inner join field_shift using(field, first_doc_id)
                            where field = ?
                            group by value, score
                            order by value, score
                        """,
                        [field],
                    )

                    db.execute(
                        """
                        INSERT into final.location_index 
                            select
                                field, 
                                value,
                                mod_location,
                                sum(location_count),
                                roaring_shift_union(passage_ids, cumulative_shift)
                            from location_index_segment
                            inner join field_shift using(field, first_doc_id)
                            where field = ?
                            group by value, mod_location
                            order by value, mod_location
                        """,
                        [field],
                    )

            db.execute(
                """
                UPDATE final.field_summary as fs
                    set
                        (unique_value_count, min_value, max_value) = (
                            select count(*), min(value), max(value) 
                            from inverted_index ii 
                            where ii.field = fs.field
                        )
                """
            )

            db.execute("commit")

        except Exception as e:
            db.execute("rollback")
            raise
