"""
Tests for type detection utilities.

Tests the KeyedData protocol and is_keyed_data function.
"""

import pandas as pd

from openleaf.performance.utils.types import is_keyed_data


def test_is_keyed_data():
    # Plain dict implements KeyedData (attrs is optional at runtime)
    d = {"a": 1, "b": 2}
    assert is_keyed_data(d)

    # pandas DataFrame implements KeyedData (has __getitem__, get, and attrs)
    df = pd.DataFrame({"altitude": [10000, 20000], "mach": [0.7, 0.8]})
    df.attrs["flight_id"] = "TEST123"
    assert is_keyed_data(df)

    # Custom class with __getitem__, get, AND attrs (full protocol)
    class CustomDataWithAttrs:
        def __init__(self):
            self.data = {"x": 10}
            self.attrs = {"metadata": "test"}

        def __getitem__(self, key):
            return self.data[key]

        def get(self, key, default=None):
            return self.data.get(key, default)

    assert is_keyed_data(CustomDataWithAttrs())

    # Custom class without attrs (still valid - attrs is optional)
    class CustomDataNoAttrs:
        def __init__(self):
            self.data = {"x": 10}

        def __getitem__(self, key):
            return self.data[key]

        def get(self, key, default=None):
            return self.data.get(key, default)

    assert is_keyed_data(CustomDataNoAttrs())

    # Plain object without methods
    class PlainObject:
        pass

    assert not is_keyed_data(PlainObject())

    # Object with only __getitem__
    class OnlyGetItem:
        def __getitem__(self, key):
            return key

    assert not is_keyed_data(OnlyGetItem())

    # Numbers and strings are not keyed data
    assert not is_keyed_data(42)
    assert not is_keyed_data("hello")
    assert not is_keyed_data([1, 2, 3])  # List has __getitem__ but not get
