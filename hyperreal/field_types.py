"""
A FieldType describes how a collection of one or more values is to be indexed.

"""

from abc import abstractmethod
from collections import Counter


class FieldType:

    @property
    @abstractmethod
    def values(self):
        return []

    @property
    def value_locations(self):
        return []

    @property
    def value_scores(self):
        return []


class ValueSequence(list, FieldType):

    def __init__(self, values):
        super().__init__(values)
        self.value_freqs = Counter(values)

    @property
    def values(self):
        return self.value_freqs.keys()

    @property
    def value_locations(self):
        return [(value, location) for location, value in enumerate(self)]

    @property
    def value_scores(self):
        return self.value_freqs.items()


class ValueSet(set, FieldType):
    def __init__(self, values):
        super().__init__(values)

    @property
    def values(self):
        return self


class Value(FieldType):
    def __init__(self, value):
        self.value = value

    @property
    def values(self):
        return [self.value]


class RangeEncodableValue(FieldType):
    def __init__(self, value):
        self.value = value

    @property
    def values(self):
        return [self.value]
