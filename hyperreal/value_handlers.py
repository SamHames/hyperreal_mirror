"""
Value handlers describe how to work with values extracted from a document in different
contexts.

These allow conversion to/from:

- the database representing the index
- HTML for display through the web interface
- strings for textual formats like CSV and for use in URLs

Shouldn't need to think too much when using builtin types: these default handlers are
already present on anything subclassed from corpus.

This is also the place for customising the rich display of fields and parts of fields,
for example for concordances, passages, or snippets. 

Note that the protocol assumes that the specific corpus is available on the
ValueHandler: this enables per corpus control of display and evaluation of types.

"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from datetime import date, datetime
from typing import Optional, Protocol
from urllib.parse import quote_plus, unquote

from tinyhtml import frag, h


class ValueHandler:
    """
    A ValueHandler describes how to transform values for use in different contexts.

    A value is an arbitrary Python object, so this is necessary to enable rich
    rendering and display in different contexts, including:

    - For storage as a value in the SQLite database representing the `index`.
    - For rendering as HTML through the web interface.
    - For rendering as a URL query parameter
    - For transforming to and from a string for CSV and when generating URLs.

    """

    value_name: str
    supported_types: set

    def __init__(self, corpus):
        self.corpus = corpus

    def from_index(self, value):
        """Create a Python object from the value stored as a single field in SQLite."""
        return value

    def to_index(self, value):
        """Transform to an SQLite compatible datatype such as text, blob or numeric."""
        return value

    def to_html(self, value) -> frag | int | float | str:
        """
        Transform for rich display in the web interface.

        Standard types like str's and numbers will be encoded and escaped by default.
        If you want the output to be rendered as HTML without escaping, the return
        value needs to be a type supported by tinyhtml for rendering:

        - a tinyhtml frag OR
        - an object supporting _repr_html_ as used in Jupyter notebooks OR
        - an object supporting __html__, as used in Jinja, MarkupSafe and some other
          templating systems.

        If you have a str you want to include as is without escaping, use
        `tinyhtml.raw`:

        > raw('should <not>be escaped</not>')

        """
        return self.to_str(value)

    def to_url(self, value) -> str:
        """
        Return a URL safe version of a string.

        """
        return quote_plus(self.to_str(value))

    def from_url(self, value):
        """
        Create a Python representation of the value from a URL string.

        """
        return self.from_str(unquote(value))

    def from_str(self, value: str):
        """Create a Python object from the string representation."""
        return value

    def to_str(self, value) -> str:
        """Create a string version of the object for terminal and other uses."""
        return str(value)


class StringHandler(ValueHandler):
    """Handles strings, by not doing anything to them anywhere."""

    value_name = "str"
    supported_types = set([str])

    def to_str(self, value: str) -> str:
        return value


class IntegerHandler(ValueHandler):
    """
    Handles integers and things convertible to integers.

    Values between -2**63 and 2**63 - 1 are supported - larger values cannot be inserted
    into an SQLite database as an integer.

    """

    value_name = "int"
    supported_types = set([int])

    def to_html(self, value):
        return str(value)

    def from_str(self, value):
        return int(value)

    def to_str(self, value):
        return str(value)


class FloatHandler(IntegerHandler):
    """
    Handles floats.

    It's likely that you will want to round these values in some way though, as
    values with only a single document are not very useful. Note also that this may not
    be completely round-trippable.

    """

    value_name = "float"
    supported_types = set([float])

    def from_str(self, value):
        return float(value)


class DateHandler(ValueHandler):
    """
    Handles dates using the inbuilt `datetime.date` type.

    Values are stored as ISO8601 strings.

    """

    value_name = "date"
    supported_types = set([date])

    def from_index(self, value):
        return date.fromisoformat(value)

    def to_index(self, value):
        return value.isoformat()

    def to_html(self, value):
        return h("time")(value.isoformat())

    def from_str(self, value):
        return date.fromisoformat(value)

    def to_str(self, value):
        return value.isoformat()


class DatetimeHandler(DateHandler):
    """
    Handles dates using the inbuilt `datetime.datetime` type.

    Values are stored as ISO8601 strings. Note that precise timezones are converted
    to UTC offsets for storage: some loss may occur.

    """

    value_name = "datetime"
    supported_types = set([datetime])

    def from_index(self, value):
        return datetime.fromisoformat(value)

    def from_str(self, value):
        return datetime.fromisoformat(value)
