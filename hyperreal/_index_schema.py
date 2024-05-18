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
CURRENT_SCHEMA_VERSION = 10

# This maps schema versions to offsets in the list of migration steps to take.
# The keys correspond to versions that have been recorded in pragma user_version;
SCHEMA_VERSION_STEPS = {0: 0, 10: 17}

MIGRATION_STEPS = [
    # 0
    "pragma application_id = 715973853",
    # 1
    """
    create table if not exists settings (
        key primary key,
        value
    )
    """,
    # 2
    """
    create table if not exists doc_key (
        doc_id integer primary key,
        doc_key unique
    )
    """,
    # 3
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
    # 4
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
    # 5
    """
    create table if not exists position_index (
        feature_id references inverted_index on delete cascade,
        first_doc_id integer,
        position_count integer,
        positions roaring_bitmap,
        primary key (feature_id, first_doc_id)
    )
    """,
    # 6
    """
    create table if not exists field_summary (
        field text primary key,
        distinct_values integer,
        min_value,
        max_value,
        position_count
    )
    """,
    # 7
    """
    create index if not exists docs_counts on inverted_index(docs_count);
    """,
    # 8
    """
    create index if not exists field_docs_counts on inverted_index(field, docs_count);
    """,
    # 9
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
    # 10
    """
    create table if not exists feature_cluster (
        feature_id integer primary key references inverted_index(feature_id) on delete cascade,
        cluster_id integer references cluster(cluster_id) on delete cascade,
        docs_count integer,
        -- Whether the feature is pinned, and shouldn't be considered for moving.
        pinned bool default 0
    )
    """,
    # 11
    """
    create index if not exists cluster_features on feature_cluster(
        cluster_id,
        docs_count
    )
    """,
    # 12
    """
    -- Used to track when clusters have changed, to mark that housekeeping
    -- functions need to run. Previously a more complex set of triggers was used,
    -- but that leads to performance issues on models with large numbers of
    -- features as triggers are only executed per row in sqlite.
    create table if not exists changed_cluster (
        cluster_id integer primary key references cluster on delete cascade
    )
    """,
    # 13
    """
    create trigger if not exists insert_feature_checks before insert on feature_cluster
        begin
            -- Make sure the cluster exists in the tracking table for foreign key relationships
            insert or ignore into cluster(cluster_id) values (new.cluster_id);
            -- Make sure that the new cluster is marked as changed so it can be summarised
            insert or ignore into changed_cluster(cluster_id) values (new.cluster_id);
        end;
    """,
    # 14
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
    # 15
    """
    create trigger if not exists delete_feature_checks before delete on feature_cluster
        begin
            -- Make sure that the new and old clusters are marked as changed
            -- so it can be summarised
            insert or ignore into changed_cluster(cluster_id)
                values (old.cluster_id);
        end;

    """,
    # 16
    "pragma user_version = 10",
]


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

    to_run = MIGRATION_STEPS[SCHEMA_VERSION_STEPS[db_version] :]

    db.execute("begin")
    try:
        for step in to_run:
            if isinstance(step, str):
                db.execute(step)
            elif callable(step):
                step(db)
            else:
                raise TypeError("Step must be a string or callable.")

        db.execute("commit")

    except Exception:
        db.execute("rollback")
        raise

    return True
