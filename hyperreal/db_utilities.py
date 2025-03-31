"""
Utility and initialisation functions for SQLite databases.

This module exposes creation of SQLite databases with standard configuration
and the setup of the necessary adapters and custom functionality to make
working with SQLite and roaring bitmaps easier.

"""

from functools import wraps
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
    conn.create_aggregate("roaring_shift_union", 2, RoaringShiftUnion)

    if row_factory:
        conn.row_factory = row_factory

    return conn


def save_bitmap(bm):
    """
    Prepare a bitmap for saving to SQLite.

    Optimisation is applied before serialisation so that
    the saved object is as compact as possible.

    """
    if isinstance(bm, pyroaring.AbstractBitMap):
        bm.shrink_to_fit()
    bm.run_optimize()
    return bm.serialize()


def load_bitmap(bm_bytes):
    """Load a bitmap object from the database as a Python object."""
    return pyroaring.BitMap.deserialize(bm_bytes)


def load_bitmap64(bm_bytes):
    """Load a bitmap object from the database as a Python object."""
    return pyroaring.BitMap64.deserialize(bm_bytes)


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


class RoaringShiftUnion:
    """
    Shift bitset by an offset, then accumulate through union.

    """

    # pylint: disable=missing-function-docstring

    def __init__(self):
        self.bitmap = pyroaring.BitMap()

    def step(self, bitmap_bytes, shift):
        bitmap = pyroaring.BitMap.deserialize(bitmap_bytes)
        self.bitmap |= bitmap.shift(shift)

    def finalize(self):
        return save_bitmap(self.bitmap)


def atomic(func):
    """
    A decorator that nests SQLite savepoints around method calls.

    This assumes that self has a .db, as initialised using a normal IndexPlugin class.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        try:
            self.db.execute(f'savepoint "{func.__name__}"')

            results = func(*args, **kwargs)

            return results

        except Exception:
            # Rewind to the previous savepoint, then release it in the finally
            # This is necessary to behave nicely whether we are operating
            # inside a larger transaction or just in autocommit mode.
            self.db.execute(f'rollback to "{func.__name__}"')
            raise

        finally:
            self.db.execute(f'release "{func.__name__}"')

    return wrapper
