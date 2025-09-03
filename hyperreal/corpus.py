"""
A HyperrealCorpus models accessing and transforming a collection of documents.

The corpus is the core point to customise hyperreal to work with the specifics of your
data.

This corpus interface is designed to enable:

- working with large collections of documents without loading everything into memory
- leaving your documents where they are
- fine-grained control over how your documents are transformed for display and indexing. 

Corpus requirements

- picklable (link to an example place where this is done with an sqlite database)
- streamable, don't hold documents in memory, encourage only holding the one document
- efficiently retrieve arbitrary sets of documents

TODO: Indexable docs format -> link to the index_builder documentation.

Links to a corpus design guide in the docs?
Links to the examples gallery.

"""

import abc
import collections
import dataclasses as dc
import mmap
import re
from typing import Any, Hashable, Iterable, Optional, TypeVar

from tinyhtml import frag, h, raw

from . import value_handlers, doc_feature_tools as dft

DocKey = TypeVar("DocKey")
Doc = TypeVar("Doc")
Value = Hashable
IndexableDoc = dict[str, list[Value] | set[Value] | Value]

default_handlers = set(
    (
        value_handlers.StringHandler,
        value_handlers.IntegerHandler,
        value_handlers.FloatHandler,
        value_handlers.DateHandler,
        value_handlers.DatetimeHandler,
    )
)


class SchemaValidationError(Exception):
    """Used for problems with creating a schema of value_handlers."""


class HyperrealCorpus:

    handler_registry: set[value_handlers.ValueHandler] = default_handlers
    extra_css = ""

    def _create_type_maps(self) -> None:
        """Create a more usable mapping between types/value names and their handlers."""

        type_schema = {}
        named_handlers = {}

        for handler_class in self.handler_registry:

            handler = handler_class(self)
            name = handler.value_name

            if name in named_handlers:
                raise SchemaValidationError(
                    f"Handler {handler} has the same `value_name` {name} as "
                    f"handler {named_handlers[name]}. Names must be unique for all "
                    "handlers defined on a corpus."
                )
            else:
                # Initialise and inject the current corpus into the handler.
                named_handlers[name] = handler

            for supported_type in handler.supported_types:

                if supported_type in type_schema:
                    raise SchemaValidationError(
                        f"Type {supported_type} can be handled by both {handler} "
                        f"and {type_schema[supported_type]}. A valid mapping requires "
                        "that there is only one handler for a given type."
                    )

                else:
                    type_schema[supported_type] = handler

        self._type_map = type_schema
        self._name_map = named_handlers

    @property
    def type_handlers(self) -> dict[Any, value_handlers.ValueHandler]:
        """
        Map of types of values emitted by `doc_to_features` to a handler.

        """
        if not hasattr(self, "_type_map"):
            self._create_type_maps()

        return self._type_map

    @property
    def name_handlers(self) -> dict[Any, value_handlers.ValueHandler]:
        """
        Map of value_names to handlers.

        """
        if not hasattr(self, "_name_map"):
            self._create_type_maps()

        return self._name_map

    @abc.abstractmethod
    def all_doc_keys(self) -> Iterable[DocKey]:
        """
        Iterates through all of the doc_keys, enumerating all of the docs that exist.

        """

        pass

    def __len__(self) -> int:
        """The number of documents in the collection."""
        return sum(1 for _ in self.all_doc_keys())

    @abc.abstractmethod
    def docs(self, doc_keys: Iterable[DocKey]) -> Iterable[tuple[DocKey, Doc]]:
        """
        Return an iterator of (doc_key, doc) pairs.

        Note that nothing is assumed about:

        1. What a document is, it can be any Python object.
        2. What to do if a doc_key is provided for a document that doesn't exist.

        """
        pass

    @abc.abstractmethod
    def doc_to_features(self, doc) -> IndexableDoc:
        """
        Transform a document into a set of features for indexing.

        This interface is likely to change in the future.

        """
        pass

    def doc_to_display_features(self, doc) -> IndexableDoc:
        """
        Transform a document to a version of features that are suitable for display.

        Only fields that need alternative handling for appropriate display need to be
        included in the output - but there does need to be a display value for every
        value on that field returned by a call to the doc_to_features method.

        This allows customised control of rendering of values - for example constructing
        concordance lines that include white space and punctuation, indexing types with
        parts-of-speech tag but not displaying them in concordances and so on.

        This default implementation only adjusts fields of string values in positional
        order - the adjustment is to add a trailing space. For non-whitespace delimited
        languages or more sophisticated tokenisation schemes you should override this.

        The display form of values must still work with the ValueHandler for that type.

        """

        doc_features = self.doc_to_features(doc)

        display_features = {}

        for field, values in doc_features.items():

            if isinstance(values, list):

                new_values = []

                for v in values:
                    if isinstance(v, str):
                        # index form = original, display form has a whitespace appended.
                        new_values.append(v + " ")
                    else:
                        new_values.append(v)

            display_features[field] = new_values

        return display_features

    def doc_to_html(self, doc, highlight_features=None) -> frag:
        """
        Transform a document into its HTML representation.

        If this is not overridden, the str representation of a document will be used
        (and escaped) as HTML.

        Takes an optional highlight_features argument - this can be used if you want to
        highlight search results within the context of the document.

        """
        return str(doc)

    def features_to_html_concordance(
        self, doc_features, display_features, highlight_features
    ) -> frag:
        """
        Convert doc features to concordance lines.

        """
        concordances = dft.to_matching_neighbourhood(
            doc_features, highlight_features, display_features=display_features
        )

        output = []

        for field, values in concordances.items():

            output.append(h("dt")(h("em")(field)))

            for match_value, match_lines in values.items():
                value_handler = self.type_handlers[type(match_value)]

                display_lines = []

                for match_position, pre, display_match, post in match_lines:

                    # This will be wrong for rtl languages - we want this ltr expected
                    # text to clip/ellipsis on what is usually the wrong side for
                    pre_html = [
                        value_handler.to_html(v.replace("\n", " ")) for v in pre
                    ]
                    match_html = h("mark")(
                        value_handler.to_html(display_match.replace("\n", " "))
                    )
                    post_html = [
                        value_handler.to_html(v.replace("\n", " ")) for v in post
                    ]

                    display_lines.append(
                        h("tr")(
                            h("td", klass="concordance-pre")(
                                h("span", dir="ltr")(pre_html)
                            ),
                            h("td", klass="concordance-match")(match_html),
                            h("td", klass="concordance-post")(post_html),
                        )
                    )

                output.append(
                    h("dd")(
                        h("details")(
                            h("summary")(
                                match_value.replace("\n", " "),
                                f" ({len(display_lines)})",
                            ),
                            h("table", klass="concordance")(display_lines),
                        )
                    )
                )

        return h("dl", klass="matches cluster")(output)

    def html_search_results(self, doc_keys, highlight_features=None) -> frag:
        """
        Create HTML fragment for search results for a specified set of documents.

        This method can be used to control everything about how documents retrieved by
        a search can be displayed, including the document itself, the elements of the
        document that matched, and concordances for matches in positional fields.

        This default implementation uses the doc_to_html converter on single documents
        and extracts concordances against matching features for each.

        You can override this method to completely customise search results. The
        interface is designed so that you can take into account as much or as little
        information as you want about the set of keys to be rendered.

        """

        results = []

        for doc_key, doc in self.docs(doc_keys):

            # Render the doc
            doc_html = self.doc_to_html(doc, highlight_features=highlight_features)

            # Extract features for concordances
            doc_features = self.doc_to_features(doc)
            display_features = self.doc_to_display_features(doc)

            highlight_features = highlight_features or []

            doc_concordances = None
            if highlight_features:
                doc_concordances = self.features_to_html_concordance(
                    doc_features, display_features, highlight_features
                )

            results.append(
                h("li", klass="search-hit")(
                    h("h3")("Document: ", doc_key), doc_html, doc_concordances
                )
            )

        return h("ul", klass="stack search-results")(results)

    def close(self) -> None:
        """
        Close the corpus object, closing any and all associated resources.

        """
        pass


boundary_regex = re.compile(r"\b")


def boundary_tokeniser(text: str) -> list[str]:
    """
    A simplistic regular expression based tokeniser.

    This tokeniser:

    - lowercases the text
    - splits on re module "word" boundaries (matches the regular expression "\b")
    - filters any token that is only whitespace (using str.strip)

    """
    return [
        stripped
        for token in boundary_regex.split(text.lower())
        if (stripped := token.strip())
    ]


def display_tokens(text: str) -> list[str]:
    """The same tokens as boundary tokeniser, but preserving case and whitespace."""


# TODO: Add a corpus for the standard folder full of text files. Possibly using the
# folder as some kind of metadata?
class TextfileParagraphsCorpus(HyperrealCorpus):

    def __init__(
        self,
        text_file_path,
        tokeniser=boundary_tokeniser,
        encoding="utf8",
        paragraph_positions=None,
    ):
        """
        A corpus that treats the paragraphs of a text file as a sequence of documents.

        Paragraph boundaries are identified as lines with only whitespace characters.

        This corpus memory maps and reads the file once from beginning to end to
        identify paragraph boundaries for efficient enumeration and random access
        later.

        """

        self.text_file_path = text_file_path
        self.encoding = encoding
        self.tokeniser = tokeniser

        self.f = open(self.text_file_path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

        self._paragraph_positions = None

    def __getstate__(self):
        return (
            self.text_file_path,
            self.tokeniser,
            self.encoding,
            self._paragraph_positions,
        )

    def __setstate__(self, args):
        self.__init__(*args)

    def all_doc_keys(self):
        return range(len(self.paragraph_positions) - 1)

    def docs(self, doc_keys):
        for key in doc_keys:

            start = self.paragraph_positions[key]
            end = self.paragraph_positions[key + 1]

            yield key, {
                "para_no": key,
                "text": self.mm[start:end].decode(self.encoding),
            }

    def doc_to_features(self, doc):
        return {"para_no": doc["para_no"], "text": self.tokeniser(doc["text"])}

    def doc_to_html(self, doc, highlight_features=None):
        return h("span")(doc["para_no"], doc["text"][:20]), h("p")(doc["text"])

    def close(self):
        self.mm.close()
        self.f.close()

    @property
    def paragraph_positions(self) -> list[int]:
        """
        A list of the start position of each paragraph as bytes in the text file.

        """

        if self._paragraph_positions is None:

            self.mm.seek(0)

            last_line_not_empty = False
            para_positions = [0]
            pos = 0

            for line in iter(self.mm.readline, b""):

                line_not_empty = bool(line.decode(self.encoding).strip())

                if line_not_empty and not last_line_not_empty:
                    para_positions.append(pos)

                pos = self.mm.tell()
                last_line_not_empty = line_not_empty

            para_positions.append(self.mm.tell())

            self._paragraph_positions = para_positions

        return self._paragraph_positions
