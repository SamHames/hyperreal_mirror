"""
Utility and initialisation functions for SQLite databases.

This module exposes creation of SQLite databases with standard configuration
and the setup of the necessary adapters and custom functionality to make
working with SQLite and roaring bitmaps easier.

"""

import sqlite3

import pyroaring


def dict_factory(cursor, row):
    """
    A factory for result rows in dictionary format with column names as keys.

    """
    # Based on the Python standard library docs:
    # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.row_factory
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def connect_sqlite(db_path, row_factory=None):
    """
    A standardised initialisation approach for SQLite.

    This connect function:

    - sets isolation_level to None, so DBAPI does not manage transactions
    - connects to the database so column type declaration parsing is active

    Note that this function setups up global adapters on the sqlite module,
    so use with care.

    """

    conn = sqlite3.connect(
        db_path, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None
    )

    conn.create_aggregate("roaring_union", 1, RoaringUnion)

    if row_factory:
        conn.row_factory = row_factory

    return conn


def save_bitmap(bm):
    """
    Prepare a bitmap for saving to SQLite.

    Optimisation is applied before serialisation so that
    the saved object is as compact as possible.

    """
    bm.shrink_to_fit()
    bm.run_optimize()
    return bm.serialize()


def load_bitmap(bm_bytes):
    """Load a bitmap object from the database as a Python object."""
    return pyroaring.BitMap.deserialize(bm_bytes)


sqlite3.register_adapter(pyroaring.BitMap, save_bitmap)
sqlite3.register_adapter(pyroaring.FrozenBitMap, save_bitmap)
sqlite3.register_converter("roaring_bitmap", load_bitmap)


class RoaringUnion:
    """
    Allows calling `roaring_union` as a function inside SQLite group by.

    """

    # pylint: disable=missing-function-docstring

    def __init__(self):
        self.bitmap = pyroaring.BitMap()

    def step(self, bitmap):
        self.bitmap |= pyroaring.BitMap.deserialize(bitmap)

    def finalize(self):
        return save_bitmap(self.bitmap)
