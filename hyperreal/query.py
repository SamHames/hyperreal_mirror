"""
Allows creating abstract queries, then evaluating them against a specific index.

In addition to allowing composing complex queries in Python, this also allows for 
serialising and displaying queries as HTML and in other formats.

"""

from __future__ import annotations

from collections import deque, abc
from typing import Protocol, runtime_checkable, TYPE_CHECKING, Sequence, Sized

from pyroaring import AbstractBitMap, BitMap

if TYPE_CHECKING:
    from .index_core import HyperrealIndex, Feature


@runtime_checkable
class Query(Protocol):

    query_type: str
    args: Sized

    # TODO: make this a fragment, and make sure it prints nicely.
    # serialize_index, html, str? Can we do something there with that?
    # Serialising to str is serialising to a querystring in a url?
    # Also that means the from_str serialisation matters :)

    # TODO: how to handle the different contexts of features?

    def evaluate(self, idx: HyperrealIndex) -> BitMap:
        pass

    def to_html(self, idx: HyperrealIndex) -> str:
        pass

    def serialize_args(self):

        # Reverse, so we can efficiently pop the (first) item off the list.
        to_serialise = list(reversed(self.args))

        if not to_serialise:
            # TODO: does an empty list make sense as a valid query?
            yield ("a", 0)

        while to_serialise:

            arg = to_serialise.pop()

            if isinstance(arg, Query):
                yield from arg.serialize()

            elif isinstance(arg, abc.Sized):
                if isinstance(arg[0], str):
                    if len(arg) == 2:
                        yield ("f", arg[0])
                        yield ("a", 1)
                        yield ("v", arg[1])
                    elif len(arg) == 3:
                        yield ("f", arg[0])
                        yield ("a", 2)
                        yield ("v1", arg[1])
                        yield ("v2", arg[2])

                # Handle arbitrary groups of features
                elif isinstance(arg[0], abc.Sized):
                    yield ("a", len(arg))
                    for subarg in reversed(arg):
                        to_serialise.append(subarg)

    def to_index_rows(self, idx: HyperrealIndex):
        """ """

        yield ("qt", self.query_type)
        yield ("a", len(self.args))
        yield from self.serialize_args()


def deserialize(idx: HyperrealIndex, index_rows):

    input_stack = list(index_rows)
    working_stack = []

    # TODO: more specific errors - probably a QueryDeserialisationError
    while input_stack:

        print(input_stack, "\n", working_stack, "\n\n")
        next_item = input_stack.pop()
        key, value = next_item

        # Values on a feature
        if key in ("v", "v1", "v2"):
            working_stack.append(next_item)

        # Arity or grouping of items
        elif key == "a":
            # Making sure we do this in the right order.
            grouped = tuple(working_stack.pop() for _ in range(value))
            working_stack.append(grouped)

        elif key == "f":
            value_args = working_stack.pop()

            # Check the length
            # Check the keys
            # Check the combinations
            if len(value_args) == 1:

                arg_name, arg_value = value_args[0]
                if arg_name != "v":
                    raise ValueError(f"expected 'v', got {arg_name}")

                # TODO: use field handler to convert index value to Python value
                working_stack.append((value, arg_value))

            elif len(value_args) == 2:
                if ("v1", "v2") != (value_args[0][0], value_args[1][0]):
                    raise ValueError(f"expected v1, v2, got {value_args}")

                working_stack.append((value, value_args[0][1], value_args[1][1]))

        elif key == "qt":
            # TODO: lookup the querytype on the idx! This is how the customisation
            # happens
            query_class = idx.defined_queries[value]
            working_stack.append(query_class(working_stack.pop()))

    print(input_stack, working_stack)
    if len(working_stack) != 1:
        raise ValueError("remaining items")

    return working_stack[0]


class MatchAny(Query):

    query_type = "match_any"

    def __init__(self, args: abc.Sized[Query | Feature]):
        self.args = args

        # TODO - validate input arguments in general?
        # This should probably only need to be in the init for each - the validation
        # will take place at each layer.
        # TODO - a bitmap is a valid query, but do we actually want to be able to
        # serialise it? Could possibly only serialise queries and features...

    def evaluate(self, idx: HyperrealIndex):

        # TODO - validate types on input, workout what other validation happens for
        # the index calls? Might make sense to push some of the feature associated
        # work down to the index.

        result = BitMap()

        for arg in self.args:

            if isinstance(arg, Query):
                result |= arg.evaluate(idx)

            elif isinstance(arg, tuple):
                result |= idx[arg][0]

        return result


class MatchAll(Query):

    query_type = "match_all"

    def __init__(self, args: abc.Sized[Query | Feature]):
        self.args = args

    def evaluate(self, idx: HyperrealIndex):

        if not self.args:
            return BitMap()

        # Initialise with the first arg
        if isinstance(self.args[0], Query):
            result = self.args[0].evaluate(idx)

        elif isinstance(self.args[0], tuple):
            result = idx[self.args[0]][0]

        for arg in self.args[1:]:

            if isinstance(arg, Query):
                result &= arg.evaluate(idx)

            elif isinstance(arg, tuple):
                result &= idx[0][0]

            # If the result set is already empty, we can return empty straight away.
            if not result:
                return result

        return result


class MatchPhrase(Query):

    query_type = "match_phrase"

    def __init__(self, args: abc.Sequence[Feature]):
        self.args = args

    def evalute(self, idx: HyperrealIndex):
        return idx.match_phrase(*args)
