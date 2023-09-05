#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 16

int main(int argc, char * argv[]){

  double * mat;
  int i, j;

  int size = 1, rank = 0, n_loc, i_g, j_g; 
  
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if( !rank ) fprintf( stdout, "Running with %d processes...\n", size);

  n_loc = N / size;
  rest = N % size;
  
  if ( rank < rest ){
    n_loc += 1;
    offset = 0;
  }
  else offset = rest; 
  
  mat = (double *) malloc( n_loc * N * sizeof(double) );

  for( i = 0; i< n_loc; i++ ){
    for( j = 0; j< N; j++ ){

      j_g = j;
      i_g = (rank * n_loc) + i + offset;
      
      if( i_g == j_g ) mat[ i * N + j ] = 1.0;
      else mat[ i * N + j ] = 0.0;
    }
  }

#ifdef PRINT_WITH_MPI_GATHER
  double * mat_g;
  if( !rank ) mat_g = (double *) malloc( N * N * sizeof(double) );
  
  MPI_Gather(mat, n_loc * N, MPI_DOUBLE, mat_g, n_loc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  if( ! rank ){

    for( i = 0; i< N; i++ ){
      for( j = 0; j< N; j++ ){
	
	fprintf( stdout, "%.3g " );
      }
      fprintf( stdout, "\n" );
    }
  }

  if( !rank ) free( mat_g );
#else

  if( !rank ){

    print_mat( );
    for( count = 1; count < size; count ++ )
      {
	MPI_Recv(mat, n_loc * N, MPI_DOUBLE, count, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	print_mat( );
      }
    
  }else{
    MPI_Send(mat, n_loc * N, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
  }

  
#end
  
  free( mat );


  MPI_Finalize();
  
  return 0;
}
      
