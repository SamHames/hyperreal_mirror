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


class ValueHandler(Protocol):
    """
    A ValueHandler describes how to transform values for use in different contexts.

    A value is an arbitrary Python object, so this is necessary to enable rich
    rendering and display in different contexts, including:

    - For storage as a value in the SQLite database representing the `index`.
    - For rendering as HTML through the web interface.
    - For transforming to and from a string for CSV and when generating URLs.

    """

    stored_sorted: bool
    value_name: str
    supported_types: set

    def from_index(self, value):
        """Create a Python object from the value stored as a single field in SQLite."""
        return value

    def to_index(self, value):
        """Transform to an SQLite compatible datatype such as text, blog or numeric."""
        return value

    def to_html(self, value) -> str:
        """Transform for rich display in the web interface."""
        return self.to_str(value)

    def from_str(self, value: str):
        """Create a Python object from the string representation."""
        return value

    def to_str(self, value) -> str:
        """Create a string version of the object for CSV and URLs."""
        return str(value)


class StringHandler(ValueHandler):
    """Handles strings, by not doing anything to them anywhere."""

    # Note that strings are lexicographically sorted, but no locale specific collation
    # is possible to apply, so it's safer to say they aren't sorted.
    stored_sorted = False
    value_name = "str"
    supported_types = set([str])

    def to_str(self, value: str) -> str:
        return value


class IntegerHandler(ValueHandler):
    """Handles integers and only things convertible to integers via `int`"""

    stored_sorted = True
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

    stored_sorted = True
    value_name = "float"
    supported_types = set([float])

    def from_str(self, value):
        return float(value)


class DateHandler(ValueHandler):
    """
    Handles dates using the inbuilt `datetime.date` type.

    Values are stored as ISO8601 strings.

    """

    stored_sorted = True
    value_name = "date"
    supported_types = set([date])

    def from_index(self, value):
        return date.fromisoformat(value)

    def to_index(self, value):
        return value.isoformat()

    def to_html(self, value):
        # TODO: should the output be a string that is treated as raw HTML always, or
        # should a str be escaped and only known HTML renderables be included as is?
        return f"<time>{value.isoformat()}</time>"

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

    stored_sorted = True
    value_name = "datetime"
    supported_types = set([datetime])

    def from_index(self, value):
        return datetime.fromisoformat(value)

    def from_str(self, value):
        return datetime.fromisoformat(value)
