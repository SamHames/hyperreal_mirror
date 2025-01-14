"""
Test value handlers.

Functionally value handlers should provide round-trips between Python, the database and
more.

"""

import math
import sqlite3

from hypothesis import given, strategies
import pytest

from hyperreal.value_handlers import (
    StringHandler,
    IntegerHandler,
    FloatHandler,
    DateHandler,
    DatetimeHandler,
)


handlers_strategies = (
    (StringHandler(), strategies.text()),
    (DateHandler(), strategies.dates()),
    (DatetimeHandler(), strategies.datetimes()),
    (IntegerHandler(), strategies.integers(min_value=-(2**63), max_value=2**63 - 1)),
    # Note that nan does round trip, but nan is by definition never equal to nan.
    (FloatHandler(), strategies.floats(allow_nan=False)),
)


@pytest.fixture(scope="module", name="db")
def fixture_db():
    """
    A preinitialised sqlite db in memory for the round trip operation.

    """
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.execute("create table x(value)")

    yield db


@pytest.mark.parametrize("handler, strategy", handlers_strategies)
@given(strategies.data())
def test_process_request(db, handler, strategy, data):
    """
    Test that the value roundtrips through SQLite correctly.

    This is necessary to ensure that things like type affinities and SQLite based limits
    that might be different from Python limits are documented correctly.

    """
    example = data.draw(strategy)

    try:
        db.execute("begin")
        db.execute("insert into x(value) values(?)", [handler.to_index(example)])

        value_from_index = list(db.execute("select * from x"))[0][0]
        assert handler.from_index(value_from_index) == example

    finally:
        db.execute("rollback")
