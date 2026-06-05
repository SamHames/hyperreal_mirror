"""
A corpus example for tabular data in a spreadsheet.

Limitations:

This isn't appropriate for large datasets: 

1. Each process will hold a complete copy of the spreadsheet and will create memory
pressure if you have a large spreadsheet
2. You can only fit ~1million rows in a spreadsheet
3. Access to rows in the spreadsheet can be slower than data formats that are more
   appropriate for high performance analytics.

"""

import openpyxl

from hyperreal.corpus import HyperrealCorpus


class SpreadsheetCorpus(HyperrealCorpus):

    def __init__(self, spreadsheet, text_columns, display_columns, index_columns):
        pass

    def all_doc_keys(self):
        pass

    def docs(self, doc_keys):
        pass

    def doc_to_features(self, doc):
        pass

    def doc_to_display_features(self, doc):
        pass

    def doc_to_html(self, doc, highlight_features=None):
        pass
