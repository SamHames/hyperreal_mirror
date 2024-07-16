"""
_index_schema.py: used for managing the schema of the index database.

Migrations are managed as a linear set of steps that are run sequentially. Schema
versions consist of one or more steps in this sequence. Migrating from an old version
to the current version requires looking up the sequence of steps up til that version
in SCHEMA_VERSION_STEPS, then running all of the steps after that one.

Each step can be either a str or a callable that takes a db connection as an argument.
Strings will be treated as SQL statements and executed as is, callables will be called
with the database connection as the only argument.


"""

# The application ID uses SQLite's pragma application_id to quickly identify index
# databases from everything else.
MAGIC_APPLICATION_ID = 715973853
CURRENT_SCHEMA_VERSION = 11

# There's one block of statements per schema_version number - each of these is basically
# the string of statements needed to migrate to the final version in the query.
# The pragma schema_version statement at the end records the actual final version.
initial_migration_steps = [
    "pragma application_id = 715973853",
    """
    create table if not exists settings (
        key primary key,
        value
    )
    """,
    """
    create table if not exists doc_key (
        doc_id integer primary key,
        doc_key unique
    )
    """,
    """
    create table if not exists inverted_index (
        feature_id integer primary key,
        field text not null,
        value not null,
        docs_count integer not null,
        doc_ids roaring_bitmap not null,
        unique (field, value)
    )
    """,
    """
    create table if not exists position_doc_map (
        field text not null,
        first_doc_id integer not null,
        last_doc_id integer not null,
        docs_count integer not null,
        doc_ids roaring_bitmap not null,
        doc_boundaries roaring_bitmap not null,
        primary key (field, first_doc_id)
    )
    """,
    """
    create table if not exists position_index (
        feature_id references inverted_index on delete cascade,
        first_doc_id integer,
        position_count integer,
        positions roaring_bitmap,
        primary key (feature_id, first_doc_id)
    )
    """,
    """
    create table if not exists field_summary (
        field text primary key,
        distinct_values integer,
        min_value,
        max_value,
        position_count
    )
    """,
    """
    create index if not exists docs_counts on inverted_index(docs_count);
    """,
    """
    create index if not exists field_docs_counts on inverted_index(field, docs_count);
    """,
    """
    -- The summary table for clusters,
    -- and the materialised results of the query and document counts.
    create table if not exists cluster (
        cluster_id integer primary key,
        feature_count integer default 0,
        -- Length of doc_ids/number of docs retrieved by the union
        docs_count integer default 0,
        -- Sum of the length of the individual feature queries that form the union
        weight integer default 0,
        doc_ids roaring_bitmap,
        -- Whether the cluster is pinned, and should be excluded from automatic clustering.
        pinned bool default 0
    );
    """,
    """
    create table if not exists feature_cluster (
        feature_id integer primary key references inverted_index(feature_id) on delete cascade,
        cluster_id integer references cluster(cluster_id) on delete cascade,
        docs_count integer,
        -- Whether the feature is pinned, and shouldn't be considered for moving.
        pinned bool default 0
    )
    """,
    """
    create index if not exists cluster_features on feature_cluster(
        cluster_id,
        docs_count
    )
    """,
    """
    -- Used to track when clusters have changed, to mark that housekeeping
    -- functions need to run. Previously a more complex set of triggers was used,
    -- but that leads to performance issues on models with large numbers of
    -- features as triggers are only executed per row in sqlite.
    create table if not exists changed_cluster (
        cluster_id integer primary key references cluster on delete cascade
    )
    """,
    """
    create trigger if not exists insert_feature_checks before insert on feature_cluster
        begin
            -- Make sure the cluster exists in the tracking table for foreign key relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);
            -- Make sure that the new cluster is marked as changed so it can be summarised
            insert or ignore into changed_cluster(cluster_id) values (new.cluster_id);
        end;
    """,
    """
    create trigger if not exists update_feature_checks before update on feature_cluster
        when old.cluster_id != new.cluster_id
        begin
            -- Make sure the new cluster exists in the tracking table for foreign
            -- key relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);

            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id)
                values (new.cluster_id), (old.cluster_id);
        end;
    """,
    """
    create trigger if not exists delete_feature_checks before delete on feature_cluster
        begin
            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id)
                values (old.cluster_id);
        end;

    """,
    "pragma user_version = 10",
]


# Remove feature IDs from the schema - only ever use concrete (field, value) pairs
# to refer to the schema.
drop_feature_ids = [
    # Migrate feature_cluster, making sure to recreate the appropriate triggers and
    # indexes afterwards.
    """
    CREATE table new_feature_cluster (
        field text,
        value,
        cluster_id integer references cluster(cluster_id) on delete cascade,
        docs_count integer,
        primary key (field, value)
    )
    """,
    """
    INSERT into new_feature_cluster
        select
            field,
            value,
            cluster_id,
            ii.docs_count
        from feature_cluster
        inner join inverted_index ii using(feature_id)
    """,
    "DROP table feature_cluster",
    "ALTER table new_feature_cluster rename to feature_cluster",
    """
    CREATE index if not exists cluster_features on feature_cluster(
        cluster_id,
        docs_count
    )
    """,
    """
    CREATE trigger if not exists insert_feature_checks before insert on feature_cluster
        begin
            -- Make sure the cluster exists in the tracking table for foreign key
            -- relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);
            -- Make sure that the new cluster is marked as changed so it can be
            -- summarised
            insert or ignore into changed_cluster(cluster_id) values (new.cluster_id);
        end;
    """,
    """
    CREATE trigger if not exists update_feature_checks before update on feature_cluster
        when old.cluster_id != new.cluster_id
        begin
            -- Make sure the new cluster exists in the tracking table for foreign
            -- key relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);

            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id)
                values (new.cluster_id), (old.cluster_id);
        end;
    """,
    """
    CREATE trigger if not exists delete_feature_checks before delete on feature_cluster
        begin
            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id)
                values (old.cluster_id);
        end;
    """,
    # Migrate positions and inverted index.
    """
    CREATE table if not exists new_inverted_index (
        field text not null,
        value not null,
        docs_count integer not null,
        doc_ids roaring_bitmap not null,
        primary key (field, value)
    ) without rowid
    """,
    """
    INSERT into new_inverted_index
        select field, value, docs_count, doc_ids
        from inverted_index
        order by field, value
    """,
    """
    CREATE table if not exists new_position_index (
        field text,
        value,
        first_doc_id integer,
        position_count integer,
        positions roaring_bitmap,
        foreign key (field, value) references new_inverted_index,
        primary key (field, value, first_doc_id)
    ) without rowid
    """,
    # pylint: disable=duplicate-code
    """
    INSERT into new_position_index
        select 
            field, 
            value,
            first_doc_id,
            position_count, 
            positions
        from position_index
        inner join inverted_index using(feature_id)
        order by field, value, first_doc_id
    """,
    # pylint: enable=duplicate-code
    "DROP table position_index",
    "ALTER table new_position_index rename to position_index",
    "DROP table inverted_index",
    "ALTER table new_inverted_index rename to inverted_index",
    "pragma user_version = 11",
]


# This maps schema versions to offsets in the list of migration steps to take.
# The keys correspond to versions that have been recorded in pragma user_version;
SCHEMA_VERSION_STEPS = {0: initial_migration_steps, 10: drop_feature_ids}


class MigrationError(ValueError):
    """Raised when a migration step fails."""


def migrate(db):
    """
    Migrate the database to the current version of the index schema.

    Returns True if a migration operation ran, False otherwise.

    """

    db_version = list(db.execute("pragma user_version"))[0][0]

    if db_version == CURRENT_SCHEMA_VERSION:
        return False

    if 0 < db_version < 10:
        raise MigrationError(
            "Migrating from this version is unsupported - please install "
            "version 0.5.0 and migrate there first."
        )

    db.execute("begin")

    while db_version != CURRENT_SCHEMA_VERSION:

        to_run = SCHEMA_VERSION_STEPS[db_version]

        try:
            for step in to_run:
                if isinstance(step, str):
                    db.execute(step)
                elif callable(step):
                    step(db)
                else:
                    raise TypeError("Step must be a string or callable.")

        except Exception:
            db.execute("rollback")
            raise

        db_version = list(db.execute("pragma user_version"))[0][0]

    db.execute("commit")

    # Only run after a migration - we can't run this during the open transaction either.
    db.execute("vacuum")

    return True
