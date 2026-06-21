"""
A corpus example for various kinds of tabular-like data.

This supports a variety of simple tabular structured formats - it allows making some
basic selections about what to treat as text and what columns to use as filters when
searching, but is otherwise quite restricted.

"""

import asyncio
import collections
import dataclasses as dc
import os
import pathlib
import random
import re
import sqlite3
import typing

from openpyxl import load_workbook
from loky import get_reusable_executor
from tinyhtml import h

from hyperreal.corpus import HyperrealCorpus
from hyperreal.field_types import ValueSequence, ValueSet
from hyperreal.index_core import HyperrealIndex, TableFilter
from hyperreal.web_server import serve_index


boundary_regex = re.compile(r"\b|\s+")
display_regex = re.compile(r"(\b|\s+)")


def tokenise(text):
    return [t for t in boundary_regex.split(text.lower()) if t]


def display_tokenise(text):
    boundaries = [t for t in display_regex.split(text) if t]

    token_start = 0

    tokens = []

    for token_idx, part in enumerate(boundaries):
        # If it doesn't match our token boundary, it's a token and needs to be appended,
        # along with any previous split boundary tokens.
        if not display_regex.fullmatch(part):
            tokens.append("".join(boundaries[token_start : token_idx + 1]))
            token_start = token_idx + 1

    return tokens


@dc.dataclass
class TabularCorpus(HyperrealCorpus):
    """
    A tabular corpus works with a wide variety of tabular formats.

    This is intended as a simple way to get started working with your data, and may not
    be suitable for all collections.

    """

    corpus_db_path: pathlib.Path | str
    text_fields: list[str]

    filter_fields: list[str] = dc.field(default_factory=list)

    header_fields: list[str] = dc.field(default_factory=list)
    inline_fields: list[str] = dc.field(default_factory=list)

    display_style: str = "document"
    display_transcript_context_turns: int = 0

    def __post_init__(self):
        self.db = sqlite3.connect(str(self.corpus_db_path), isolation_level=None)
        self.db.execute(
            """
            CREATE table if not exists doc (
                doc_id integer,
                field text,
                value
            )
            """
        )

    def __getstate__(self):

        data = {k: v for k, v in self.__dict__.items() if k != "db"}

        return data

    def __setstate__(self, data):
        self.__dict__ = data
        self.__post_init__()

    def replace_rows(self, row_field_values):
        """
        Delete all current docs and replace them with the provided field values.

        """
        self.db.execute("BEGIN")
        self.db.execute("DROP table if exists doc")
        self.db.execute(
            """
            CREATE table doc (
                doc_id integer,
                field text,
                value
            )
            """
        )

        self.db.executemany(
            "INSERT INTO doc values (?, ?, ?)",
            (
                (doc_id, field or "", value)
                for doc_id, field_values in enumerate(row_field_values)
                for field, value in field_values
            ),
        )

        self.db.execute("CREATE index doc_by_id on doc(doc_id, field)")
        self.db.execute("COMMIT")

    def replace_rows_from_parquet(self, parquet_path):

        worksheet = spreadsheet[sheetname]

        rows = worksheet.values
        header = next(rows)

        replace_rows = (dict(zip(header, row)) for row in rows)

        self.replace_rows(replace_rows)

    def replace_rows_from_spreadsheet(self, spreadsheet, sheetname):

        worksheet = spreadsheet[sheetname]

        rows = worksheet.values
        header = next(rows)

        replace_rows = (zip(header, row) for row in rows)

        self.replace_rows(replace_rows)

    def __len__(self):
        return list(self.db.execute("SELECT max(doc_id) + 1 from doc"))[0][0]

    def all_doc_keys(self):
        for (doc_id,) in self.db.execute("SELECT distinct doc_id from doc"):
            yield doc_id

    def docs(self, doc_keys):

        for doc_id in doc_keys:

            field_values = collections.defaultdict(list)

            for field, value in self.db.execute(
                "SELECT field, value from doc where doc_id = ?", [doc_id]
            ):
                field_values[field].append(value)

            yield doc_id, field_values

    def doc_to_features(self, doc):

        features = {}

        for field, values in doc.items():

            if field in self.text_fields:
                features[field] = ValueSequence(
                    [
                        tok
                        for val in values
                        if val is not None
                        for tok in tokenise(val)
                        if tok
                    ]
                )

            else:
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    features[field] = ValueSet(valid_values)

        return features

    def render_text_with_inline_fields(self, doc, highlight_features=None):

        display_text = {field: " ".join(doc[field]) for field in self.text_fields}

        if highlight_features:
            highlight_fields = set(f[0] for f in highlight_features)

            if highlight_fields & set(self.text_fields):

                for field in self.text_fields:
                    match_features = {
                        value for f, value in highlight_features if field == f
                    }

                    tokens = tokenise(" ".join(doc.get(field, [])))
                    display_tokens = display_tokenise(" ".join(doc.get(field, [])))

                    display_text[field] = h("span")(
                        (
                            display_token
                            if token not in match_features
                            else h("mark")(display_token)
                        )
                        for token, display_token in zip(tokens, display_tokens)
                    )

        display = []

        for text_component in display_text.values():

            inline = None
            if self.inline_fields:
                inline = h("span", klass="inline-fields")(
                    doc.get(inline_field, []) for inline_field in self.inline_fields
                )
            display.append(h("div")(inline, text_component))

        return display

    def render_header(self, doc):

        return h("ul", klass="cluster result-header")(
            (h("span")(val) for val in doc[field]) for field in self.header_fields
        )

    def html_search_results(self, doc_ids, highlight_features=None):

        search_hits = []

        for doc_id in doc_ids:

            turn_context_range = [doc_id]

            if self.display_style == "transcript":
                turn_context_range = range(
                    max(0, doc_id - self.display_transcript_context_turns),
                    min(doc_id + self.display_transcript_context_turns + 1, len(self)),
                )

            docs = list(self.docs(turn_context_range))

            for key, doc in docs:
                if key == doc_id:
                    break

            header = self.render_header(doc)

            components = [
                self.render_text_with_inline_fields(
                    d, highlight_features=highlight_features
                )
                for _, d in docs
            ]

            search_hits.append(h("li", klass="search-hit stack")(header, *components))

        return h("ul", klass="stack search-results")(search_hits)

    def features_to_html_concordance(
        self, doc_features, display_features, highlight_features
    ):
        return None

    extra_css = """
        .result-header {
            font-weight: bold;
        }

        .search-results {
            font-family: monospace, monospace;
        }

        .inline-fields {
            font-style: italic;
            float: left;
        }

        .inline-fields::after {
            content: ":";
            margin-right: var(--s-2);
        }
    """


def serve_tabular_corpus(corpus):

    pool = get_reusable_executor()

    tabular_idx = HyperrealIndex("corpus_index.db", corpus, pool)

    print("Indexing documents")
    tabular_idx.rebuild()

    tabular_idx.search_fields = {field: tokenise for field in corpus.text_fields}

    # Configure the interface facets and search
    tabular_idx.facets = [
        (
            field,
            tabular_idx.field_features(field, min_docs=1),
            TableFilter(order_by="hits", first_k=20, keep_above=0),
        )
        for field in corpus.filter_fields
    ]

    print("Creating feature clusters")

    clustering = tabular_idx.plugins["feature_clusters"]

    # Set the state of the RNG to a consistent point.
    tabular_idx.random_state = random.Random(42)

    # Initialise with a random clustering
    random_clustering = clustering.initialise_random_clustering(
        64, min_docs=5, include_fields=corpus.text_fields
    )

    clustering.replace_clusters(random_clustering)

    # Refine the clustering for a small number of iterations - we could go for
    # longer, but it usually doesn't matter as you'll spend the same amount of time
    # examining the output either way.
    clustering.refine_clustering(iterations=50)

    jupyter_hub_service_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    url_base = jupyter_hub_service_prefix + "proxy/absolute/9999"

    loop = asyncio.get_running_loop()
    task = loop.create_task(serve_index(tabular_idx, base_path=url_base))

    display_link = h("a", href=url_base + "/browse/")("Browse your table")
    display(display_link)
