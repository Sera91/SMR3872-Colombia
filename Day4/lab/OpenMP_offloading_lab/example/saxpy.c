#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)

/* 
  SAXPY OpenMP offload with different options (TEST)
*/
void saxpy_gpu_test(float *x, float*y, float a, int sz, int nteams) {
  float t = 0.0;
  double tb, te;
  tb = omp_get_wtime();
  // #pragma omp target map(to:x[0:sz]) map(tofrom:y[0:sz])
  // #pragma omp teams loop
  // #pragma omp parallel for simd
  // #pragma omp target loop map(to:x[0:sz]) map(tofrom:y[0:sz])
  for (int i = 0; i < sz; i++) {
    y[i] = a * x[i] + y[i];
  }
  te = omp_get_wtime();
  t = te - tb;
  printf("Time of kernel: %lf\n", t);
}

/* 
  SAXPY OpenMP offload with nteams defined (see commented lines
  for other options. 
*/
void saxpy_gpu_nteams(float *x, float*y, float a, int sz, int nteams) {
  float t = 0.0;
  double tb, te;
  tb = omp_get_wtime();
  #pragma omp target teams distribute parallel for simd \
        num_teams(nteams) map(to:x[0:sz]) map(tofrom:y[0:sz])
  for (int i = 0; i < sz; i++) {
    y[i] = a * x[i] + y[i];
  }
  te = omp_get_wtime();
  t = te - tb;
  printf("Time of kernel nteams %d: %lf\n", nteams, t);
}

/* 
  SAXPY OpenMP offload with nteams auto 
*/
void saxpy_gpu_nteams_auto(float *x, float*y, float a, int sz) {
  float t = 0.0;
  double tb, te;
  tb = omp_get_wtime();
  #pragma omp target teams distribute parallel for simd \
        map(to:x[0:sz]) map(tofrom:y[0:sz])
  for (int i = 0; i < sz; i++) {
    y[i] = a * x[i] + y[i];
  }
  te = omp_get_wtime();
  t = te - tb;
  printf("Time of kernel nteams auto: %lf\n", t);
}

int main (int argc, char *argv[]){

  long dim = 0;
  int nteams;
  float *x, *y, a;
  double tb, te;

  printf("There are %d devices\n", omp_get_num_devices());

  // Check on input parameters
  if(argc != 2) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim \n");
    return 1;
  }

  dim = strtol(argv[1], NULL, 10);

  printf("Matrix dimension: %d\n", dim);

  // Initializes arrays
  x = malloc(sizeof(float) * dim);
  y = malloc(sizeof(float) * dim);

  // Populate arrays
  tb = omp_get_wtime();
  #pragma omp parallel for
  for( int i = 0 ; i < dim ; i++ ) {
    x[i] = 1.0/(i*i);
    y[i] = x[i]*x[i];
  }
  te = omp_get_wtime();

  printf("Time populating arrays: %lf\n", (te - tb));
  
  // Populate scalar
  a = AM;

  // Benchmark nteams size
  for (int i = 0; i < 10; i++) {
    nteams = pow(2, i);
    saxpy_gpu_nteams(x, y, a, dim, nteams);  
  }

  // Let OMP decide nteams
  saxpy_gpu_nteams_auto(x, y, a, dim);   
  
  free(x);
  free(y);

  return(0);

}

