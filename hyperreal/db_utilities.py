"""
Utility and initialisation functions for SQLite databases.

This module exposes creation of SQLite databases with standard configuration
and the setup of the necessary adapters and custom functionality to make
working with SQLite and roaring bitmaps easier.

"""

import sqlite3
from functools import wraps

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
    A decorator that ensures an atomic sqlite transaction for a group of operations.

    This assumes that self has a .db and a .idx attribute, as initialised using a normal
    IndexPlugin class.

    This decorator will also call the post_transaction method on self when succesfully
    completing a full set of nested transactions.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]

        # This could be used on the core, or a plugin.
        if hasattr(self, "idx"):
            idx = self.idx
        else:
            idx = self

        try:

            # Only start a transaction on the first level
            # Note that we're using savepoints here to work within the context of an
            # explicit index transaction defined by the with idx: construct.
            if not idx._transaction_level:
                idx.db.execute(f"savepoint start")

            idx._transaction_level += 1
            results = func(*args, **kwargs)

            return results

        except Exception as e:
            idx._transaction_error = True
            raise

        finally:

            # Work out how to end the transaction once we've unrolled everything.
            # Note that we don't decrement the transaction level until after we've
            # done the finalisation, so any calls to post_transaction know we're in
            # a transaction.
            if idx._transaction_level == 1:

                if idx._transaction_error:
                    idx.db.execute("rollback to start")
                else:
                    for plugin in idx.plugins.values():
                        plugin.post_transaction()

                    idx.db.execute(f"release start")

                idx._transaction_level = 0
                idx._transaction_error = False

            else:
                idx._transaction_level -= 1

    return wrapper
