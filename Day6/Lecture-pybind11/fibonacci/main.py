import time
import fibonacci_example as fibex

def fibonacci_py(x):
    if x < 2:
        return x
    return fibonacci_py(x - 1) + fibonacci_py(x - 2)

n = 40

print('Python:')
start_time = time.perf_counter_ns()
print('Answer:', fibonacci_py(n))
print('Time:', (time.perf_counter_ns() - start_time) / 1e9, 's')
print()

print('C++:')
start_time = time.perf_counter_ns()
print('Answer:', fibex.fibonacci_cpp(n))
print('Time:', (time.perf_counter_ns() - start_time) / 1e9, 's')
