import test
import pytest

import env
from pybind11_tests import IncType, UserType
from pybind11_tests import builtin_casters as bc


#TESTING BUILT-IN TYPE CASTING

def test_int_long():
    """In Python 2, a C++ int should return a Python int rather than long
    if possible: longs are not always accepted where ints are used (such
    as the argument to sys.exit()). A C++ long long is always a Python
    long."""

    import sys

    must_be_long = type(getattr(sys, "maxint", 1) + 1)
    assert isinstance(bc.int_cast(), int)
    assert isinstance(bc.long_cast(), int)
    assert isinstance(bc.longlong_cast(), must_be_long)



def test_bytes_to_string():
    """Tests the ability to pass bytes to C++ string-accepting functions.  Note that this is
    one-way: the only way to return bytes to Python is via the pybind11::bytes class."""
    # Issue #816

    def to_bytes(s):
        b = s if env.PY2 else s.encode("utf8")
        assert isinstance(b, bytes)
        return b

    assert bc.strlen(to_bytes("hi")) == 2


#TESTING OPERATOR OVERLOADING
from pybind11_tests import operators as m


def test_operator_overloading():
    v1 = m.Vector2(1, 2)
    v2 = m.Vector2(2, 3)  
    assert v1 is not v2

    assert str(v1) == "[1.000000, 2.000000]"

    assert str(-v1) == "[-1.000000, -2.000000]"

    assert str(v1 + v2) == "[3.000000, 5.000000]"
    assert str(v1 - v2) == "[-1.000000, -1.000000]"







