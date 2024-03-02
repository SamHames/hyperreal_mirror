"""
Test at the CLI layer - these function closer to end to end tests and
should test many of the most common entrypoints.

"""

import pathlib

from click.testing import CliRunner

from hyperreal import cli

data_path = pathlib.Path("tests", "data")
corpora_path = pathlib.Path("tests", "corpora")


def test_plaintext_corpus(tmp_path):
    """Basic tests of the plaintext corpus support."""
    target_corpora_db = tmp_path / "test.db"
    target_index_db = tmp_path / "test_index.db"
    target_graphml = tmp_path / "test.graphml"
    target_csv = tmp_path / "test.csv"

    runner = CliRunner()

    # Create
    result = runner.invoke(
        cli.plaintext_corpus,
        [
            "create",
            str(data_path / "alice30.txt"),
            str(target_corpora_db),
        ],
    )
    assert result.exit_code == 0

    # Index
    result = runner.invoke(
        cli.plaintext_corpus,
        [
            "index",
            str(target_corpora_db),
            str(target_index_db),
        ],
    )
    assert result.exit_code == 0

    args = [
        "--iterations",
        "10",
        "--clusters",
        "10",
        "--min-docs",
        "10",
        str(target_index_db),
    ]

    # Model - empty case
    result = runner.invoke(cli.model, args)

    assert result.exit_code == 0

    # Repeat the model, should error as a model already exists
    result = runner.invoke(cli.model, args + ["--restart"])

    assert result.exit_code == 1

    # Restart the model with confirmation
    result = runner.invoke(
        cli.model,
        args + ["--restart"],
        input="Y",
    )

    assert result.exit_code == 0

    # Graph
    result = runner.invoke(
        cli.export,
        ["graph", str(target_index_db), str(target_graphml)],
    )

    assert result.exit_code == 0

    # Graph
    result = runner.invoke(
        cli.export,
        [
            "graph",
            str(target_index_db),
            str(target_graphml),
            "--exclude-field-in-label",
        ],
    )

    assert result.exit_code == 0

    # Clusters

    result = runner.invoke(
        cli.export,
        ["clusters", str(target_index_db), str(target_csv)],
    )

    assert result.exit_code == 0

    # Export sample

    result = runner.invoke(
        cli.plaintext_corpus,
        [
            "sample",
            str(target_corpora_db),
            str(target_index_db),
            str(target_csv),
            "--docs-per-cluster=10",
        ],
    )

    assert result.exit_code == 0


def test_sx_corpus(tmp_path):
    """Basic tests of the stackexchange corpus support via the CLI."""
    target_corpora_db = tmp_path / "sx_corpus.db"
    target_index_db = tmp_path / "sx_corpus_index.db"
    target_csv = tmp_path / "sx_sample.csv"

    runner = CliRunner()

    # Add site
    result = runner.invoke(
        cli.stackexchange_corpus,
        [
            "add-site",
            str(data_path / "expat_sx" / "Posts.xml"),
            str(data_path / "expat_sx" / "Comments.xml"),
            str(data_path / "expat_sx" / "Users.xml"),
            "https://expatriates.stackexchange.com",
            str(target_corpora_db),
        ],
    )
    assert result.exit_code == 0

    # Index
    result = runner.invoke(
        cli.stackexchange_corpus,
        ["index", str(target_corpora_db), str(target_index_db)],
    )
    assert result.exit_code == 0

    # Model
    result = runner.invoke(
        cli.model,
        [
            "--iterations",
            "1",
            "--clusters",
            "16",
            "--min-docs",
            "10",
            "--include-field",
            "Post",
            "--include-field",
            "UserPosting",
            str(target_index_db),
        ],
    )

    assert result.exit_code == 0

    # Export sample

    result = runner.invoke(
        cli.stackexchange_corpus,
        [
            "sample",
            str(target_corpora_db),
            str(target_index_db),
            str(target_csv),
            "--docs-per-cluster=10",
        ],
    )

    assert result.exit_code == 0
