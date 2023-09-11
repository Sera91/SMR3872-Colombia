#ifndef FIBONACCI
#define FIBONACCI


unsigned int Fibonacci_cpp(const unsigned int n){

    if (n < 2){
        return n;
    }
    return Fibonacci_cpp(n - 1) + Fibonacci_cpp(n - 2);
}

#endif
