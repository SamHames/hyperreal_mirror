"""
Value handlers describe how to work with the values extracted from each field in a doc.

These allow conversion to/from:

- the database representing the index
- HTML for display through the web interface
- strings for textual formats like CSV and for use in URLs

This is also the place for customising the rich display of fields and parts of fields,
for example for concordances, passages, or snippets. 

"""

from collections.abc import Hashable, Sequence
from datetime import date, datetime
from html import escape
from typing import Optional, Protocol

from markupsafe import Markup


class ValueHandler(Protocol):
    """
    A ValueHandler describes how to transform values for use in different contexts.

    A value is an arbitrary Python object, so this is necessary to enable rich
    rendering and display in different contexts, including:

    - For storage as a value in the SQLite database representing the `index`.
    - For rendering as HTML through the web interface.
    - For transforming to and from a string for CSV and when generating URLs.

    """

    def from_index(self, value):
        """Create a Python object from the value stored as a single field in SQLite."""
        return value

    def to_index(self, value) -> Hashable:
        """Transform to an SQLite compatible datatype such as text, blog or numeric."""
        return value

    def to_html(self, value):
        """Transform for rich display in the web interface."""
        return self.to_str(value)

    def from_str(self, value: str):
        """Create a Python object from the string representation."""
        return value

    def to_str(self, value) -> str:
        """Create a string version of the object for CSV and URLs."""
        return str(value)


class SupportsSegment(Protocol):
    """
    A ValueHandler can implement these fields to display parts of documents.

    This is optional and intended for usecases like displaying parts of document through
    concordances or retrieval of passages.
    """

    def segment_to_str(
        self, values, start, end, highlight: Optional[Sequence] = None
    ) -> str:
        """
        Take a segment of a sequence of values and render it to a single string.

        This is used to create passages and concordances from the output of
        `corpus.doc_to_features`.

        """
        selected_values = values[start:end]

        if highlight is not None:
            highlight = set(highlight)
            selected_values = [
                f"**{value}**" if value in highlight else value
                for value in selected_values
            ]

        return " ".join(selected_values)

    def segment_to_html(self, values, start, end, highlight: Optional[Sequence] = None):
        """
        Take a segment of a sequence of values and render it to a single HTML string.

        This is used to create passages and concordances from the output of
        `corpus.doc_to_features`.

        """
        selected_values = values[start:end]

        if highlight is not None:
            highlight = set(highlight)
            selected_values = [
                f"<mark>{escape(value)}</mark>" if value in highlight else value
                for value in selected_values
            ]

        return Markup(" ".join(selected_values))


class NoopHandler(ValueHandler):
    """A simple handler that does nothing but pass values through."""


class StringHandler(SupportsSegment, ValueHandler):
    """Everything is saved as a string, and otherwise kept unchanged."""

    def to_index(self, value):
        return str(value)

    def to_str(self, value):
        return str(value)


class IntegerHandler(ValueHandler):
    """Handles integers and only things convertible to integers via `int`"""

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
    values with only a single document are not very useful.

    """

    def from_str(self, value):
        return float(value)


class DateHandler(ValueHandler):
    """
    Handles dates using the inbuilt `datetime.date` type.

    Values are stored as ISO8601 strings.

    """

    def from_index(self, value):
        return date.fromisoformat(value)

    def to_index(self, value):
        return value.isoformat()

    def to_html(self, value):
        return Markup(f"<time>{value.isoformat()}</time>")

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

    def from_index(self, value):
        return datetime.fromisoformat(value)

    def from_str(self, value):
        return datetime.fromisoformat(value)
