#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NN 4096
#define NM 4096

double A[NN][NM];
double Anew[NN][NM];

int main() {

    int n = NN;
    int m = NM;
    int iter_max = 100;
    double tol = 1.0e-6;

    double error = 1.0;
    int iter = 0;
    int i,j;
    struct timeval temp_1, temp_2;
    double ttime=0.;

    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

    for (j = 0; j < n; j++)
    {
        A[j][0]    = 1.0;
        Anew[j][0] = 1.0;
    }

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    gettimeofday(&temp_1, (struct timezone*)0);

    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

        for( j = 1; j < n-1; j++) {
            for( i = 1; i < m-1; i++ ) {
                Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
                                    + A[j-1][i] + A[j+1][i]);
                error = fmax( error, fabs(Anew[j][i] - A[j][i]));
            }
        }

        for( j = 1; j < n-1; j++) {
            for( i = 1; i < m-1; i++ ) {
                A[j][i] = Anew[j][i];
            }
        }

        if(iter % 10 == 0) printf("%5d, %0.8lf\n", iter, error);
        iter++;
    }

    gettimeofday(&temp_2, (struct timezone*)0);
    ttime = 0.000001*((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec));

    printf("Elapsed time (s) = %.2lf\n", ttime);
    printf("Stopped at iteration: %u\n", iter);

    return 0;

}
