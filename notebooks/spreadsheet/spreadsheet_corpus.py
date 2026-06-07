"""
A corpus example for tabular data in a spreadsheet.

Limitations:

This isn't appropriate for large datasets:

1. Each process will hold a complete copy of the spreadsheet and will create memory
pressure if you have a large spreadsheet
2. You can only fit ~1million rows in a spreadsheet
3. Access to rows in the spreadsheet can be slower than data formats that are more
   appropriate for high performance analytics.


TODO: Shouldn't I just converge this to an SQLite database instead?

"""

import collections
import dataclasses as dc
import pathlib
import typing

from openpyxl import load_workbook

from hyperreal.corpus import HyperrealCorpus
from hyperreal.field_types import ValueSequence, Value, RangeEncodableValue


def extract_header(sheet) -> list[str]:
    """
    Extract the contiguous header columns from the first row of the sheet.

    """
    first_row = list(sheet.iter_rows(min_row=1, max_row=1, values_only=True))

    headers = []

    if first_row:

        for val in first_row[0]:

            if isinstance(val, str):
                headers.append(val)
            else:
                break

    return headers


def identify_spreadsheet_tables(
    spreadsheet_path: pathlib.Path | str,
) -> dict[str, list[str]]:
    """
    Identify tables, columns and candidate joins in a give spreadsheet.

    """

    sheet_columns = dict()

    wb = load_workbook(spreadsheet_path)

    for sheet in wb:

        sheet_name = sheet.title

        headers = extract_header(sheet)

        if headers:
            sheet_columns[sheet_name] = extract_header(sheet)

    join_check = [(name, set(columns)) for name, columns in sheet_columns.items()]

    join_candidates = collections.defaultdict(dict)

    for i, (i_name, i_columns) in enumerate(join_check):

        for j_name, j_columns in join_check[i + 1 :]:

            inter_columns = i_columns & j_columns

            if inter_columns:
                join_candidates[i_name][j_name] = inter_columns
                join_candidates[j_name][i_name] = inter_columns

    return sheet_columns, join_candidates


sheet_columns, join_candidates = identify_spreadsheet_tables(
    "combined_transcripts_2026-06-05.xlsx"
)

print(sheet_columns)

for table, join_columns in join_candidates.items():
    for join_table, columns in join_columns.items():
        print(table, join_table, columns)


@dc.dataclass
class XslxCorpus(HyperrealCorpus):

    spreadsheet_path: pathlib.Path | str
    text_sheet: str
    text_columns: list[str]
    tokeniser: typing.Optional = None
    display_tokeniser: typing.Optional = None

    docs: list[tuple] = dc.field(init=False)
    text_col_idx: list[int] = dc.field(init=False)
    header_idx: list[tuple] = dc.field(init=False)

    def __post_init__(self):

        wb = load_workbook(self.spreadsheet_path)

        sheet = wb[self.text_sheet]

        header = extract_header(sheet)
        self.header_idx = {col: i for i, col in enumerate(header)}

        all_rows = sheet.values
        # Skip the header
        next(all_rows)

        for col in self.text_columns:
            if col not in self.header_idx:
                raise ValueError(f"Text column {col} not found in sheet {text_sheet}")

        # We'll store the docs as a simple in memory list for now. There's only so many
        # documents we can fit in a spreadsheet after all.
        self.docs = []

        for row in all_rows:
            self.docs.append(row)

        # Next step: pull out the other tables as little in memory lookups based on the
        # selected keys.

    def __len__(self):
        return len(self.docs)

    def all_doc_keys(self):
        return range(len(self.docs))

    def docs(self, doc_keys):
        for key in doc_keys:
            yield key, self.docs[key]

    def doc_to_features(self, doc):
        return {
            col: ValueSequence((doc[self.header_idx[col]] or "").split())
            for col in self.text_columns
        }

    def html_search_results(self, doc_keys, highlight_features=None):
        # This will be where all the search display magic happens. We want customisable
        # display of context as well, and we don't want concordances for this initial
        # use case, so we'll do it with this.
        pass

    def features_to_html_concordance(
        self, doc_features, display_features, highlight_features
    ) -> frag:
        return None


XslxCorpus(
    spreadsheet_path="combined_transcripts_2026-06-05.xlsx",
    text_sheet="turn",
    text_columns=["transcription"],
)
