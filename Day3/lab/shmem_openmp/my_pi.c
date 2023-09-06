#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>

#define N 1000000000

#include <time.h>
#include <sys/time.h>

double seconds()
/* Returns elepsed seconds past from the last call to timer rest */
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *) 0 );
  sec = tmp.tv_sec + ( (double) tmp.tv_usec ) / 1000000.0;
  return sec;
}


int main( int argc, char * argv[] )
{

  int n = N;
  double w = 1.0 / n;
  double x, f_x, sum = 0.0;

  double t1 = 0.0, t2 = 0.0;
  int n_threads = 1;
  
  fprintf( stdout, "Running with %d processes...\n", n_threads);

  t1 = seconds();
  
  for( int i = 1 ; i <= n; i++ ) {
    x = w * (i - 0.5);
    f_x = 4.0 * 1.0 /(x*x + 1.0);
    sum += f_x;
  }

  t2 = seconds();
  
  fprintf( stdout, "The value of PI is %.10g Vs %.10g\nTime to solution %.3g (sec.)\n", sum * w );
  
  return 0;
}
  
