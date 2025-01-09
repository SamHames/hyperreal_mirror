"""
A hyperreal corpus models accessing and transforming a collection of documents.

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

Indexable docs format -> link to the index_builder documentation.

Links to a corpus design guide in the docs?
Links to the examples gallery.

Example:

"""

import abc
from typing import Protocol, Hashable, Iterable, Any, TypeVar

from tinyhtml import h

from . import value_handlers


DocKey = TypeVar("DocKey")
Doc = TypeVar("Doc")
Value = Hashable
IndexableDoc = dict[str, list[Value] | set[Value] | Value]

default_handlers = set(
    (
        value_handlers.StringHandler(),
        value_handlers.IntegerHandler(),
        value_handlers.FloatHandler(),
        value_handlers.DateHandler(),
        value_handlers.DatetimeHandler(),
    )
)


class SchemaValidationError(Exception):
    """Used for problems with creating a schema of value_handlers."""


class Corpus(Protocol):

    handler_registry: set[value_handlers.ValueHandler] = default_handlers

    def _create_type_maps(self) -> None:
        """Create a more usable mapping between types/value names and their handlers."""

        type_schema = {}
        named_handlers = {}

        for handler in self.handler_registry:

            name = handler.value_name

            if name in named_handlers:
                raise SchemaValidationError(
                    f"Handler {handler} has the same `value_name` {name} as "
                    f"handler {named_handlers[name]}. Names must be unique for all "
                    "handlers defined on a corpus."
                )
            else:
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
        Map of types of values emitted by `indexable_docs` to a handler.

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
    def indexable_docs(
        self, doc_keys: Iterable[DocKey]
    ) -> Iterable[tuple[DocKey, IndexableDoc]]:
        pass

    def html_docs(self, doc_keys: Iterable[DocKey]) -> Iterable[tuple[DocKey, str]]:
        """
        Iterate through pairs of doc_keys and the associated doc's HTML representation.

        """
        for key, str_doc in self.str_docs(doc_keys):
            yield key, h("p")(str_doc).render()

    def str_docs(self, doc_keys: Iterable[DocKey]) -> Iterable[tuple[DocKey, str]]:
        """
        Iterate through pairs of doc_keys and the associated doc's str representation.

        """
        for key, doc in self.docs(doc_keys):
            yield key, str(doc)

    def close(self) -> None:
        """
        Close the corpus object, closing any and all associated resources.

        """
        pass
