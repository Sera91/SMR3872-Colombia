#Instructions to build this package

This directory contain a simple pybind11 example code
to performe the estimation of the nth number in the
Fibonacci sequence using C++ code.


To build the package using the 'build\_ext' directive,
use the command:

$ pip install -e . -vvv

#Notes on 'build\_ext' procedure
The command 'build\_ext' builds C/C++ extension modules. It creates a command line for running the compiler and linker by combining compiler and linker options from various sources:

  - the sysconfig variables CC, CXX, CCSHARED, LDSHARED, and CFLAGS,
  - the environment variables CC, CPP, CXX, LDSHARED and CFLAGS, CPPFLAGS, LDFLAGS,
  - the Extension attributes (include\_dirs, library\_dirs, extra\_compile\_args, extra\_link\_args, runtime\_library\_dirs.)

Specifically, if the environment variables CC, CPP, CXX, and LDSHARED are set, they will be used instead of the sysconfig variables of the same names.

The compiler options appear in the command line in the following order:

    1) the options provided by the sysconfig variable CFLAGS,

    2) the options provided by the environment variables CFLAGS and CPPFLAGS,

    3) the options provided by the sysconfig variable CCSHARED,

    4) a -I option for each element of Extension.include_dirs,

    5) the options provided by Extension.extra_compile_args.

The linker options appear in the command line in the following order:

    1) the options provided by environment variables and sysconfig variables,

    2)  a -L option for each element of Extension.library_dirs,

    3) a linker-specific option like -Wl,-rpath for each element of Extension.runtime_library_dirs,

    finally, the options provided by Extension.extra_link_args.

The resulting command line is then processed by the compiler and linker. According to the GCC manual sections on directory options and environment variables, the C/C++ compiler searches for files named in #include <file> directives in the following order:

    1) in directories given by -I options (in left-to-right order),

    2) in directories given by the environment variable CPATH (in left-to-right order),

    3) in directories given by -isystem options (in left-to-right order),

    4) in directories given by the environment variable C_INCLUDE_PATH (for C) and CPLUS_INCLUDE_PATH (for C++),

    5) in standard system directories,

    6)  in directories given by -idirafter options (in left-to-right order).

The linker searches for libraries in the following order:

    1) in directories given by -L options (in left-to-right order),

    2) in directories given by the environment variable LIBRARY_PATH (in left-to-right order).

