"""
Allows creating abstract queries, then evaluating them against a specific index.

In addition to allowing composing complex queries in Python, this also allows for 
serialising and displaying queries as HTML and in other formats.

"""

from typing import Protocol, runtime_checkable

from pyroaring import AbstractBitMap, BitMap

from .index_core import HyperrealIndex, FieldValues


@runtime_checkable
class Query(Protocol):
    # TODO: make this a fragment, and make sure it prints nicely.
    def evaluate(self, idx: HyperrealIndex) -> BitMap:
        pass

    def to_html(self, idx: HyperrealIndex) -> str:
        pass

    # serialize_index, html, str? Can we do something there with that?
    # Serialising to str is serialising to a querystring in a url?
    # Also that means the from_str serialisation matters :)
    def serialize(self, idx: HyperrealIndex):
        pass

    def deserialize(self, idx: HyperrealIndex):
        pass


class MatchAny(Query):

    def __init__(self, *args: Query | FieldValues | AbstractBitMap):
        self.args = args

        # TODO - validate input arguments in general?
        # This should probably only need to be in the init for each - the validation
        # will take place at each layer.

    def evaluate(self, idx: HyperrealIndex):

        # TODO - validate types on input, workout what other validation happens for
        # the index calls? Might make sense to push some of the feature associated
        # work down to the index.

        result = BitMap()

        for arg in self.args:

            if isinstance(arg, Query):
                result |= arg.evaluate(idx)

            if isinstance(arg, AbstractBitMap):
                result |= arg

            elif isinstance(arg, FieldValues):
                result |= idx.match_any(arg)

        return result


class MatchAll(Query):

    def __init__(self, *args: Query | FieldValues | AbstractBitMap):
        self.args = args

    def evaluate(self, idx: HyperrealIndex):

        if not self.args:
            return BitMap()

        # Initialise with the first arg
        if isinstance(self.args[0], Query):
            result = self.args[0].evaluate(idx)

        elif isinstance(self.args[0], FieldValues):
            result = idx.match_all(self.args[0])

        elif isinstance(self.args[0], AbstractBitMap):
            result = self.args[0]

        for arg in self.args[1:]:

            if isinstance(arg, Query):
                result &= arg.evaluate(idx)

            elif isinstance(arg, FieldValues):
                result &= idx.match_all(arg)

            elif isinstance(arg, AbstractBitMap):
                result &= arg

        return result


class MatchPhrase(Query):
    def __init__(self, field_phrase: FieldValues):
        self.field_phrase = field_phrase

    def evalute(self, idx: HyperrealIndex):
        return idx.match_phrase(self.field_phrase)
