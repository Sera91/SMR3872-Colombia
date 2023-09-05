#include <stdlib.h>
#include <stdio.h>

#define N 16

int main(int argc, char * argv[]){

  double * mat;
  int i, j;
  
  mat = (double *) malloc( N * N * sizeof(double) );

  for( i = 0; i< N; i++ ){
    for( j = 0; j< N; j++ ){

      if( i == j ) mat[ i * N + j ] = 1.0;
      else mat[ i * N + j ] = 0.0;

      fprintf( stdout, "%.3g ", mat[ i * N + j ] );
    }
    fprintf( stdout, "\n");
  }

  free( mat );

  return 0;
}
      
