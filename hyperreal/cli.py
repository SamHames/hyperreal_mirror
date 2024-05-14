"""
This module provides all of the CLI functionality for hyperreal.

"""

import concurrent.futures as cf
import csv
import logging
import multiprocessing as mp

import click

import hyperreal.corpus
import hyperreal.index
import hyperreal.server

logging.basicConfig(level=logging.INFO)


def make_two_file_indexer(corpus_type):
    """
    Return a click annotated function for an indexer that takes two arguments.

    The expected arguments are (in order):

    1. The path to the file representing the corpus.
    2. The path to the index database.

    """

    @click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
    @click.argument("index_db", type=click.Path(dir_okay=False))
    @click.option(
        "--doc-batch-size",
        type=int,
        default=DEFAULT_DOC_BATCH_SIZE,
        help="""
            The size of individual batches of documents sent for indexing.
            Larger sizes will require more ram, but might be more efficient
            for large collections. This setting interacts with workers: each
            worker handles doc-batch-size documents, so more workers will
            consume more memory.
        """,
    )
    @click.option(
        "--index-positions",
        default=False,
        type=bool,
        is_flag=True,
        help="""
        Turn on to enable indexing of positional information of features in
        sequence fields.
        """,
    )
    @click.option(
        "--workers",
        default=8,
        type=int,
        help="""
            The number of background workers to use during indexing. Each
            worker handles handles a batch of documents, so doubling workers
            will consume double the memory. You shouldn't use more workers
            than you have CPU cores available and there will be diminishing
            returns for more workers.
        """,
    )
    def corpus_indexer(corpus_db, index_db, doc_batch_size, index_positions, workers):
        """
        Creates the index database representing the given corpus.

        If the index already exists it will be reindexed.

        """
        click.echo(f"Indexing {corpus_db} into {index_db}.")

        doc_corpus = corpus_type(corpus_db)

        mp_context = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(workers, mp_context=mp_context) as pool:
            doc_index = hyperreal.index.Index(index_db, corpus=doc_corpus, pool=pool)

            doc_index.rebuild(
                doc_batch_size=doc_batch_size, index_positions=index_positions
            )

    return corpus_indexer


@click.group(name="hyperreal")
def cli():
    """The main entry command for the whole CLI."""


@cli.group()
def plaintext_corpus():
    """Entry command for all functionality related to a plaintext corpus."""


DEFAULT_DOC_BATCH_SIZE = 5000


@plaintext_corpus.command(name="create")
@click.argument("text_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("corpus_db", type=click.Path(dir_okay=False))
def plaintext_corpus_create(text_file, corpus_db):
    """
    Create a simple corpus database from a plain text input file.

    The input file should be a plain text file, with one document per line. If
    a document begins and ends with a quote (") character, it will be treated
    as a JSON string and escapes decoded.

    If the corpus exists already, the content will be replaced with the
    contents of the text file.

    """
    click.echo(f"Replacing existing contents of {corpus_db} with {text_file}.")

    doc_corpus = hyperreal.corpus.PlainTextSqliteCorpus(corpus_db)

    with open(text_file, "r", encoding="utf-8") as infile:
        f = csv.reader(infile)
        # The only documents we drop are lines that are only whitespace.
        docs = (line[0] for line in f if line and line[0].strip())
        doc_corpus.replace_docs(docs)


@plaintext_corpus.command(name="serve")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
def plaintext_corpus_serve(corpus_db, index_db):
    """
    Serve the given plaintext corpus and index via the webserver.

    """

    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    click.echo(f"Serving corpus '{corpus_db}'/ index '{index_db}'.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(
            index_db,
            corpus_class=hyperreal.corpus.PlainTextSqliteCorpus,
            corpus_args=[corpus_db],
            pool=pool,
        )
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()


plaintext_corpus.command(name="index")(
    make_two_file_indexer(hyperreal.corpus.PlainTextSqliteCorpus)
)


@cli.group()
def stackexchange_corpus():
    """Entrypoint for all StackExchange related functionality."""


@stackexchange_corpus.command(name="replace-sites")
@click.argument(
    "archive_files",
    type=click.Path(exists=True, dir_okay=False),
    nargs=-1,
)
@click.argument("corpus_db", type=click.Path(dir_okay=False))
def stackexchange_corpus_add_site(archive_files, corpus_db):
    """
    Create a simple corpus database from the stackexchange 7z/XML data dumps.

    The data dumps for all sites can be found here:
    https://archive.org/download/stackexchange

    Dumps from multiple sites can be added to the same corpus in one go.
    The site is inferred from the filename.

    """
    doc_corpus = hyperreal.corpus.StackExchangeCorpus(corpus_db)
    doc_corpus.replace_sites_data(*archive_files)


@stackexchange_corpus.command(name="serve")
@click.argument("corpus_db", type=click.Path(exists=True, dir_okay=False))
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
def stackexchange_corpus_serve(corpus_db, index_db):
    """
    Serve the given StackExchange corpus and index via the webserver.

    """

    if not hyperreal.index.Index.is_index_db(index_db):
        raise ValueError(f"{index_db} is not a valid index file.")

    click.echo(f"Serving corpus '{corpus_db}'/ index '{index_db}'.")

    mp_context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(mp_context=mp_context) as pool:
        index_server = hyperreal.server.SingleIndexServer(
            index_db,
            corpus_class=hyperreal.corpus.StackExchangeCorpus,
            corpus_args=[corpus_db],
            pool=pool,
        )
        engine = hyperreal.server.launch_web_server(index_server)
        engine.block()


stackexchange_corpus.command(name="index")(
    make_two_file_indexer(hyperreal.corpus.StackExchangeCorpus)
)


@cli.command()
@click.argument("index_db", type=click.Path(exists=True, dir_okay=False))
@click.option("--iterations", default=10, type=click.INT)
@click.option(
    "--clusters",
    default=None,
    type=click.INT,
    help="The number of clusters to use in the model. "
    "Ignored unless this is the first run, or --restart is passed. "
    "If not provided in those cases it will default to 64. ",
)
@click.option("--min-docs", default=100)
@click.option(
    "--include-field",
    default=[],
    help="""A field to include in the model initialisation. Multiple fields
    can be specified - if no fields are provided all available fields will be
    included. Ignored unless this is the first run, or --restart is
    passed.""",
    multiple=True,
)
@click.option(
    "--restart", default=False, is_flag=True, help="Restart the model from scratch."
)
@click.option(
    "--tolerance",
    default=0.01,
    type=click.FloatRange(0, 1),
    help="Specify an early termination tolerance on the fraction of features moving. "
    "If fewer than this fraction of features moves in an iteration, the "
    "refinement will terminate early.",
)
@click.option(
    "--move-acceptance-probability",
    default=0.5,
    type=click.FloatRange(0, 1),
    help="The random chance of actually executing a possible move. "
    "Smaller values will take more iterations to converge, large "
    "values might be unstable.",
)
@click.option(
    "--random-seed",
    default=None,
    type=int,
    help="Specify a random seed for a model run. Best used with restart",
)
@click.option(
    "--minimum-cluster-features",
    default=1,
    type=click.INT,
    help="The minimum number of features in a cluster.",
)
@click.option(
    "--group-test-batches",
    default=None,
    type=click.INT,
    help="The number of grouped clusters to use to accelerate the clustering. "
    "Set to 0 to disable the group hierarchy cluster optimisation. Leaving "
    "at the default of None sets this to 0.1*clusters.",
)
@click.option(
    "--group-test-top-k",
    default=2,
    type=click.INT,
    help="The number of top groups to investigate if group-test-batches is enabled. "
    "Testing more groups will take longer but find better solutions.",
)
def model(
    index_db,
    iterations,
    clusters,
    min_docs,
    restart,
    include_field,
    random_seed,
    tolerance,
    move_acceptance_probability,
    minimum_cluster_features,
    group_test_batches,
    group_test_top_k,
):
    """
    Create or refine a new feature cluster model on the given index.

    Note that n_clusters can be changed arbitrarily, even when not initialising a new
    model with --restart.

    """
    doc_index = hyperreal.index.Index(
        index_db, hyperreal.corpus.EmptyCorpus(), random_seed=random_seed
    )

    # Check if any clusters exist.
    has_clusters = bool(doc_index.cluster_ids)

    if has_clusters:
        if restart:
            click.confirm(
                "A model already exists on this index, do you want to delete it?",
                abort=True,
            )

            # If the number of clusters isn't explicitly set.
            clusters = clusters or 64
            click.echo(
                f"Restarting new feature cluster model with {clusters} clusters on {index_db}."
            )
            doc_index.initialise_clusters(
                n_clusters=clusters,
                min_docs=min_docs,
                include_fields=include_field or None,
            )
    else:
        # If the number of clusters isn't explicitly set.
        clusters = clusters or 64
        click.echo(
            f"Creating new feature cluster model with {clusters} clusters on {index_db}."
        )
        doc_index.initialise_clusters(
            n_clusters=clusters,
            min_docs=min_docs,
            include_fields=include_field or None,
        )

    click.echo(f"Refining for {iterations} iterations on {index_db}.")
    doc_index.refine_clusters(
        iterations=iterations,
        target_clusters=clusters,
        tolerance=tolerance,
        move_acceptance_probability=move_acceptance_probability,
        minimum_cluster_features=minimum_cluster_features,
        group_test_batches=group_test_batches,
        group_test_top_k=group_test_top_k,
    )
