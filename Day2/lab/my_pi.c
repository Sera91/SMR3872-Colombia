#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 1000

int main( int argc, char * argv[] )
{

  int n = N;
  double w = 1.0 / n;
  double x, f_x, sum = 0.0;
  
  for( int i = 1; i <= n; i++ ) {

    x = w * (i - 0.5);
    f_x = 4.0 * 1.0 /(x*x + 1.0);
    sum += f_x;
  }

  fprintf( stdout, "The value of PI is %.10g Vs %.10g", sum * w, M_PI );

  return 0;
}
  
