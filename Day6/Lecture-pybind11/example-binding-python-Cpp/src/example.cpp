#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/table.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include "arrow_conversion.hh"
#include <iostream>
#include <string>
#include <arrow/array/builder_primitive.h>
#include "caster.hh"

namespace py = pybind11;


namespace {

using DoubleArray = arrow::DoubleArray;
using Array = arrow::Array;

std::shared_ptr<arrow::Table> load_table_from_csv(std::string csv_filename) {
  auto input_file =
      arrow::io::ReadableFile::Open(csv_filename.c_str()).ValueOrDie();
  auto csv_reader =
      arrow::csv::TableReader::Make(arrow::io::default_io_context(), input_file,
                                    arrow::csv::ReadOptions::Defaults(),
                                    arrow::csv::ParseOptions::Defaults(),
                                    arrow::csv::ConvertOptions::Defaults())
          .ValueOrDie();
  auto table = csv_reader->Read().ValueOrDie();
  return std::move(table);
}

py::object example_load_csv(std::string csv_filename) {
  arrow::py::import_pyarrow();
  PyObject* object = arrow::py::wrap_table(load_table_from_csv(csv_filename));
  return py::reinterpret_steal<py::object>(object);
}

//definition of vector addition, element by element
std::shared_ptr<arrow::DoubleArray> vadd(std::shared_ptr<arrow::DoubleArray>& a,
                                        std::shared_ptr<arrow::DoubleArray>& b,
                                        std::shared_ptr<arrow::DoubleArray>& c) {
    //checking input arrays have same length
    if ((a->length() != b->length()) || (a->length() != c->length())) {
        throw std::length_error("Arrays are not of equal length");
    }
    //calling the double array builder
    arrow::DoubleBuilder builder;
    arrow::Status status = builder.Resize(a->length());
    if (!status.ok()) {
        throw std::bad_alloc();
    }
    for(int i = 0; i < a->length(); i++) {
        builder.UnsafeAppend(a->Value(i) + b->Value(i) + c->Value(i));
    }
    std::shared_ptr<arrow::DoubleArray> array;
    arrow::Status st = builder.Finish(&array);
    return array;
}

//definition of vector multiplication, element by element
std::shared_ptr<arrow::DoubleArray> vmul(std::shared_ptr<arrow::DoubleArray>& a,
                                        std::shared_ptr<arrow::DoubleArray>& b,
                                        std::shared_ptr<arrow::DoubleArray>& c) {
    //checking input arrays have same length
    if ((a->length() != b->length()) || (a->length() != c->length())) {
        throw std::length_error("Arrays are not of equal length");
    }
    //calling the double array builder
    arrow::DoubleBuilder builder;
    arrow::Status status = builder.Resize(a->length());
    if (!status.ok()) {
        throw std::bad_alloc();
    }
    for(int i = 0; i < a->length(); i++) {
        builder.UnsafeAppend(a->Value(i) * b->Value(i) * c->Value(i));
    }
    std::shared_ptr<arrow::DoubleArray> array; //output array declaration
    arrow::Status st = builder.Finish(&array); //return results of builder into array
    return array;
}


}  // namespace

void print_table(std::shared_ptr<arrow::Table> &table)
{
    // print table
    std::cout << "Table schema: " << std::endl;
    std::cout << table->schema()->ToString() << std::endl;
    //std::cout << "Table columns: " << std::endl;
    //for (int i = 0; i < table->num_columns(); i++)
    //{
    //    std::cout << "Column " << i << ": " << std::endl;
    //    std::cout << table->column(i)->ToString() << std::endl;
    //}
}


//function equivalent to np.sum(array)

double sum(std::shared_ptr<arrow::DoubleArray> a) {
    double sum = 0;
    for(int i = 0; i < a->length(); i++) {
        sum += a->Value(i);
    }
    return sum;
}


//function to perform element-wise multiplication plus addiction
double madd(int a, double b, double c){
	return (a*b)+c;
}


PYBIND11_MODULE(arrow_pybind_example, m) {
  arrow::py::import_pyarrow();
  m.def("vectorized_madd", py::vectorize(madd));
  m.def("print_table", &print_table, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("vadd", &vadd, py::call_guard<py::gil_scoped_release>());
  m.def("vmul", &vmul, py::call_guard<py::gil_scoped_release>());
  m.def("sum", &sum, py::call_guard<py::gil_scoped_release>());
  m.def("example_load_csv", &example_load_csv,
        R"pbdoc(
        Loads a CSV file as a PyArrow table.
    )pbdoc");
  m.attr("__version__") = "0.1.0";
}
