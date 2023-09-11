# Apache Arrow / CMake / pybind11 Example

An example package that demonstrates:
    
- C++ extensions to Apache Arrow built with CMake
- python bindings to those extensions with pybind11.
- dependencies installed with conda

## Usage

Create and activate the conda environment:

```bash
conda env create -f environment.yml \
&& conda activate example-apache-arrow-pybind11-cmake
```

Install the python package:
```
pip install -e .
```

Test that the build was successful:
```
pytest tests
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

This example uses a modified version of the `setup.py` from the [pybind11 CMake example](https://github.com/pybind/cmake_example) that is licensed under a [BSD-style license](https://github.com/pybind/cmake_example/blob/master/LICENSE).