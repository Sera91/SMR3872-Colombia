import pytest
import pyarrow as pa
import numpy as np



def test_vadd():
    import arrow_pybind_example as ape
    a = pa.array([1, 2, 3], type=pa.float64())
    b = pa.array([0, 2, 4], type=pa.float64())
    c = pa.array([1, 1, 5], type=pa.float64())
    assert ape.vadd(a, b, c).to_pylist() == [2, 5, 12]

    with pytest.raises(ValueError):
        assert ape.vadd(a[1:], b, c).to_pylist() == [2, 5, 12]


def test_vmul():
    import arrow_pybind_example as ape
    a = pa.array([1, 2, 3], type=pa.float64())
    b = pa.array([0, 2, 4], type=pa.float64())
    c = pa.array([1, 1, 5], type=pa.float64())
    assert ape.vmul(a, b, c).to_pylist() == [0, 4, 60]

    with pytest.raises(ValueError):
        assert ape.vmul(a[1:], b, c).to_pylist() == [0, 4, 60]


def test_sum():
    import arrow_pybind_example as ape
    x = pa.array([1,2,3.1])
    assert ape.sum(x) == np.sum(np.array(x))
    assert ape.sum(x[1:]) == np.sum(np.array(x[1:]))


def test_madd():
    import arrow_pybind_example as ape
    x = np.array([1,2,3])
    y = np.array([4.,6.,5.])
    z = 7.0
    assert  ape.vectorized_madd(x, y, z).tolist() == [11.0,19.0,22.0]
