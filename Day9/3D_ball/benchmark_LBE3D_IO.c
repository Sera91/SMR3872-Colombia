#include <math.h>
#include <time.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include "hdf5.h"

#define N_STEPS 20
#define N_DIMS 3
#define STRING_SIZE 2048
#define IDX( i, j, k ) ( k + ( j + i * ( NYP2 ) ) * ( NZP2 ) );

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

// variables defined global for LBE3D compatibility 
int NPX, NPY, NPZ;
int LXS, LYS, LZS;
int NX, NY, NZ;
int NXP2, NYP2, NZP2; 
MPI_Comm MPI_COMM_CART, MPI_COMM_ALONG_X_TEMP, MPI_COMM_ALONG_Y_TEMP, MPI_COMM_ALONG_Z_TEMP;

MPI_Datatype MPI_X_RhoPlane, MPI_Y_RhoPlane, MPI_Z_RhoPlane;

MPI_Datatype MPI_Rhotype;

/* MPI variables */
int npes, mype, mype_cart[3];
// mype in all directions
int mex, mey, mez;
// neighbor process_id in all directions
int pxm, pxp, pym, pyp, pzm, pzp;

//my_coords in the processes grid
int my_coords[3];

double seconds()
/* Returns elepsed seconds past from the last call to timer rest */
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *) 0 );
  sec = tmp.tv_sec + ( (double) tmp.tv_usec ) / 1000000.0;
  return sec;
}

// Definition of the 2D grid of processes 
void define_process_grid( int *grid_global_dimensions ){

  int dims[3]; // process grid dimensions
  int coords[3], periods[3] = { 1, 1, 1 };

  switch( npes ){
    
  case 1:
    dims[ 0 ] = 1;
    dims[ 1 ] = 1;
    dims[ 2 ] = 1;
    break;

  case 2:
    dims[ 0 ] = 2;
    dims[ 1 ] = 1;
    dims[ 2 ] = 1;
    break;

  case 4:
    dims[ 0 ] = 4;
    dims[ 1 ] = 1;
    dims[ 2 ] = 1;
    break;

  case 8:
    dims[ 0 ] = 4;
    dims[ 1 ] = 2;
    dims[ 2 ] = 1;
    break;

  case 16:
    dims[ 0 ] = 4;
    dims[ 1 ] = 4;
    dims[ 2 ] = 1;
    break;

  case 32:
    dims[ 0 ] = 4;
    dims[ 1 ] = 8;
    dims[ 2 ] = 1;
    break;

  case 64:
    dims[ 0 ] = 4;
    dims[ 1 ] = 16;
    dims[ 2 ] = 1;
    break;

  case 128:
    dims[ 0 ] = 4;
    dims[ 1 ] = 32;
    dims[ 2 ] = 1;
    break;

  case 256:
    dims[ 0 ] = 4;
    dims[ 1 ] = 64;
    dims[ 2 ] = 1;
    break;

  case 512:
    dims[ 0 ] = 4;
    dims[ 1 ] = 128;
    dims[ 2 ] = 1;
    break;

  case 1024:
    dims[ 0 ] = 4;
    dims[ 1 ] = 256;
    dims[ 2 ] = 1;
    break;

  case 2048:
    dims[ 0 ] = 8;
    dims[ 1 ] = 256;
    dims[ 2 ] = 1;
    break;

  case 4096:
    dims[ 0 ] = 64;
    dims[ 1 ] = 64;
    dims[ 2 ] = 1;
    break;
    
  default:
    fprintf( stderr, "\nFatal error: nuber of processes not supported. Program aborted ..." );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }
  
  MPI_Cart_create( MPI_COMM_WORLD, 3, dims, periods, 0, &MPI_COMM_CART );
  
  /* Build the sub-communicators along X and Y */
  coords[ 0 ] = 1;
  coords[ 1 ] = 0;
  coords[ 2 ] = 0;
  MPI_Cart_sub( MPI_COMM_CART, coords, &MPI_COMM_ALONG_X_TEMP );
  coords[ 0 ] = 0;
  coords[ 1 ] = 1;
  coords[ 2 ] = 0;
  MPI_Cart_sub( MPI_COMM_CART, coords, &MPI_COMM_ALONG_Y_TEMP );
  coords[ 0 ] = 0;
  coords[ 1 ] = 0;
  coords[ 2 ] = 1;
  MPI_Cart_sub( MPI_COMM_CART, coords, &MPI_COMM_ALONG_Z_TEMP );

  /* Rank along X, Y and Z directions */
  MPI_Comm_rank( MPI_COMM_ALONG_Z_TEMP, &mez );
  MPI_Comm_rank( MPI_COMM_ALONG_Y_TEMP, &mey );
  MPI_Comm_rank( MPI_COMM_ALONG_X_TEMP, &mex );

  /*! tasks per direction, ONLY for cartesian grid */
  NPX = dims[0];
  NPY = dims[1];
  NPZ = dims[2];

  /*! size of tasks, ONLY for cartesian grid */
  if( grid_global_dimensions[ 0 ] % NPX ){
    fprintf( stderr, "\nFatal error NX = %d not multiple of the number of processes among the X dimension (%d) the process grid. Program aborted ...", grid_global_dimensions[ 0 ], NPX );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  if( grid_global_dimensions[ 1 ] % NPY ){
    fprintf( stderr, "\nFatal error NY = %d not multiple of the number of processes among the Y dimension (%d) the process grid. Program aborted ...", grid_global_dimensions[ 1 ], NPY );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  if( grid_global_dimensions[ 2 ] % NPZ ){
    fprintf( stderr, "\nFatal error NZ = %d not multiple of the number of processes among the Z dimension (%d) the process grid. Program aborted ...", grid_global_dimensions[ 2 ], NPZ );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }

  NX = grid_global_dimensions[ 0 ] / NPX;
  NY = grid_global_dimensions[ 1 ] / NPY;
  NZ = grid_global_dimensions[ 2 ] / NPZ;

  /*! logical mapping of neighbour tasks ONLY for cartesian grid (using mpi functions) */
  MPI_Cart_shift (MPI_COMM_ALONG_X_TEMP, 0, 1, &pxm, &pxp);
  MPI_Cart_shift (MPI_COMM_ALONG_Y_TEMP, 0, 1, &pym, &pyp);
  MPI_Cart_shift (MPI_COMM_ALONG_Z_TEMP, 0, 1, &pzm, &pzp);

  /*! physical coordinates of mpi tasks ONLY for cartesian grid */
  LXS = mex * NX;
  LYS = mey * NY;
  LZS = mez * NZ;

  int rank;
  MPI_Comm_rank( MPI_COMM_CART, &rank );
  MPI_Cart_coords( MPI_COMM_CART, rank, 3, my_coords );

  MPI_Type_contiguous (1, MPI_DOUBLE, &MPI_Rhotype);
  MPI_Type_commit (&MPI_Rhotype);

  MPI_Type_vector (NY, NZ, NZP2, MPI_Rhotype, &MPI_X_RhoPlane);
  MPI_Type_vector (NXP2, NZ, NZP2 * NYP2, MPI_Rhotype, &MPI_Y_RhoPlane);
  MPI_Type_vector (NXP2 * NYP2, 1, NZP2, MPI_Rhotype, &MPI_Z_RhoPlane);
  MPI_Type_commit (&MPI_Z_RhoPlane);
  MPI_Type_commit (&MPI_Y_RhoPlane);
  MPI_Type_commit (&MPI_X_RhoPlane);
}

// This function initialize a density sphere on a 3D grid distributed among a 2D grid of processes
void init_sphere( double * rho, int * grid_global_dimensions ){

  int i, j, k;
  double x0 = grid_global_dimensions[ 0 ] / 2.0, y0 = grid_global_dimensions[ 1 ] / 2.0, z0 = grid_global_dimensions[ 2 ] / 2.0; 
  for( i = 1; i <= NX; i++ ){
    for( j = 1; j <= NY; j++ ){
      for(k = 1; k <= NZ; k++ ){
	
	size_t idx = IDX( i, j, k );
	 
	size_t global_i = i + LXS;
	size_t global_j = j + LYS;
	size_t global_k = k + LZS;	
       
	double delta_x = ( global_i - x0 );
	double delta_y = ( global_j - y0 );
	double delta_z = ( global_k - z0 );
	
	if( sqrt( ( delta_x ) * (delta_x ) + ( delta_y ) * ( delta_y ) + ( delta_z ) * ( delta_z ) ) < grid_global_dimensions[ 0 ]/4.0 ){
	  rho[ idx ] = 0.8;
	} else{
	  rho[ idx ] = 0.2;
	}
      }
    }
  }
}

void evolve_sphere( double * rho, int * grid_global_dimensions, int istep ){

  int i, j, k;
  double x0 = grid_global_dimensions[ 0 ] / 2.0, y0 = grid_global_dimensions[ 1 ] / 2.0, z0 = grid_global_dimensions[ 2 ] / 2.0; 
  for( i = 1; i <= NX; i++ ){
    for( j = 1; j <= NY; j++ ){
      for(k = 1; k <= NZ; k++ ){
	
	size_t idx = IDX( i, j, k );
	 
	size_t global_i = (i + istep + LXS) % grid_global_dimensions[ 0 ];
	size_t global_j = j + LYS;
	size_t global_k = k + LZS;	
       
	double delta_x = ( global_i - x0 );
	double delta_y = ( global_j - y0 );
	double delta_z = ( global_k - z0 );
	
	if( sqrt( ( delta_x ) * (delta_x ) + ( delta_y ) * ( delta_y ) + ( delta_z ) * ( delta_z ) ) < grid_global_dimensions[ 0 ]/4.0 ){
	  rho[ idx ] = 0.8;
	} else{
	  rho[ idx ] = 0.2;
	}
      }
    }
  }
}

// This function checks if any of the readed values is = 0.0
void check_zerose( double * rho, int * grid_global_dimensions ){

  int i, j, k;
  for( i = 1; i <= NX; i++ ){
    for( j = 1; j <= NY; j++ ){
      for(k = 1; k <= NZ; k++ ){
	
	size_t idx = IDX( i, j, k );
	if( rho[ idx ] == 0.0 ){ 
	  fprintf( stderr, "\nI am %d and I found a 0.0. The program will be aborted...", mype);
	  MPI_Abort( MPI_COMM_WORLD, 1 );
	}
      }
    }
  }
}

void pbc(double *field)
{
  MPI_Status status1;

  MPI_Sendrecv( field + NZP2*NYP2*NX + NZP2+1 ,  1, MPI_X_RhoPlane, pxp, 20,
		field                + NZP2+1 ,  1, MPI_X_RhoPlane, pxm, 20, MPI_COMM_ALONG_X_TEMP, &status1);
  MPI_Sendrecv( field + NZP2*NYP2        + NZP2+1 ,  1, MPI_X_RhoPlane, pxm, 21,
		field + NZP2*NYP2*(NX+1) + NZP2+1 ,  1, MPI_X_RhoPlane, pxp, 21, MPI_COMM_ALONG_X_TEMP, &status1);

  MPI_Sendrecv( field + NZP2*(NY) + 1   ,  1, MPI_Y_RhoPlane, pyp, 22,
		field + 1               ,  1, MPI_Y_RhoPlane, pym, 22, MPI_COMM_ALONG_Y_TEMP, &status1);
  MPI_Sendrecv( field + NZP2 + 1        ,  1, MPI_Y_RhoPlane, pym, 23,
		field + NZP2*(NY+1) + 1 ,  1, MPI_Y_RhoPlane, pyp, 23, MPI_COMM_ALONG_Y_TEMP, &status1);

  MPI_Sendrecv( field + NZ   ,  1, MPI_Z_RhoPlane, pzp, 24,
		field        ,  1, MPI_Z_RhoPlane, pzm, 24, MPI_COMM_ALONG_Z_TEMP, &status1);
  MPI_Sendrecv( field + 1    ,  1, MPI_Z_RhoPlane, pzm, 25,
		field + NZ+1 ,  1, MPI_Z_RhoPlane, pzp, 25, MPI_COMM_ALONG_Z_TEMP, &status1);

}


void write_rho( int step, double * rho, int *grid_global_dimensions )
{
  char fname[ STRING_SIZE ];

  hid_t efilespace;
  hid_t plist_id;               /* property list identifier */
  hid_t file_id, group_id,dataset_id;  // identifiers 
  hid_t xfer_plist;
  herr_t ret;
  herr_t status;
  hid_t edataset, ememspace;
  hid_t hdf5_status;

  hsize_t edimens_3d[ N_DIMS ];
  hsize_t estart_3d[ N_DIMS ], ecount_3d[ N_DIMS ], estride_3d[ N_DIMS ], eblock_3d[ N_DIMS ];
  hsize_t dstart_3d[ N_DIMS ], dcount_3d[ N_DIMS ], dstride_3d[ N_DIMS ], dblock_3d[ N_DIMS ];

  double t_start = seconds();

  // Init Par HDF5
  H5Eset_current_stack (H5E_DEFAULT);

  plist_id = H5Pcreate (H5P_FILE_ACCESS);
  hdf5_status = H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  edimens_3d[ 0 ] = NXP2;
  edimens_3d[ 1 ] = NYP2;
  edimens_3d[ 2 ] = NZP2;

  ememspace = H5Screate_simple( N_DIMS, edimens_3d, NULL );

  edimens_3d[ 0 ] = grid_global_dimensions[ 0 ];
  edimens_3d[ 1 ] = grid_global_dimensions[ 1 ];
  edimens_3d[ 2 ] = grid_global_dimensions[ 2 ];

  efilespace = H5Screate_simple( N_DIMS, edimens_3d, NULL );

  estart_3d[ 0 ] = 1;
  estart_3d[ 1 ] = 1;
  estart_3d[ 2 ] = 1;
  estride_3d[ 0 ] = 1;
  estride_3d[ 1 ] = 1;
  estride_3d[ 2 ] = 1;
  eblock_3d[ 0 ] = NX;
  eblock_3d[ 1 ] = NY;
  eblock_3d[ 2 ] = NZ;
  ecount_3d[ 0 ] = 1;
  ecount_3d[ 1 ] = 1;
  ecount_3d[ 2 ] = 1;

  dstart_3d[ 0 ] = LXS;
  dstart_3d[ 1 ] = LYS;
  dstart_3d[ 2 ] = LZS;
  dstride_3d[ 0 ] = 1;
  dstride_3d[ 1 ] = 1;
  dstride_3d[ 2 ] = 1;
  dblock_3d[ 0 ] = NX;
  dblock_3d[ 1 ] = NY;
  dblock_3d[ 2 ] = NZ;
  dcount_3d[ 0 ] = 1;
  dcount_3d[ 1 ] = 1;
  dcount_3d[ 2 ] = 1;

  status = H5Sselect_hyperslab( ememspace, H5S_SELECT_SET, estart_3d, estride_3d, ecount_3d, eblock_3d );
  status = H5Sselect_hyperslab( efilespace, H5S_SELECT_SET, dstart_3d, dstride_3d, dcount_3d, dblock_3d);

  sprintf(fname,"density_t.%d.h5", step);
  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

  dataset_id = H5Dcreate(file_id, "/rho1", H5T_NATIVE_DOUBLE, efilespace, H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

  xfer_plist = H5Pcreate (H5P_DATASET_XFER);
  ret = H5Pset_dxpl_mpio (xfer_plist, H5FD_MPIO_COLLECTIVE);

  double t_start_writing = seconds();
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, ememspace, efilespace, xfer_plist, rho);

  double t_end = seconds();

  if( mype == 0 ){

    fprintf( stdout, "\n\n\tStep = %d", step );
    fprintf( stdout, "\n\tTime of write_rho = %.3g (seconds).", t_end - t_start );
    fprintf( stdout, "\n\tTime of H5Dwrite = %.3g (seconds).", t_end - t_start_writing );
    fprintf( stdout, "\n\tMeasured bandwidth of H5Dwrite = %.3g (MB/s).", ( (double) sizeof(double) * grid_global_dimensions[ 0 ] * grid_global_dimensions[ 1 ] * grid_global_dimensions[ 2 ] ) / 1000000 / ( t_end - t_start_writing ) );    
  }

  status = H5Pclose( plist_id );
  status = H5Pclose( xfer_plist );
  status = H5Dclose(dataset_id);
  status = H5Fclose(file_id);
}

void read_rho( int step, double * rho, int *grid_global_dimensions )
{
  char fname[ STRING_SIZE ];

  hid_t efilespace;
  hid_t plist_id;               /* property list identifier */
  hid_t file_id, dataset_id;  // identifiers 
  hid_t xfer_plist;
  herr_t ret;
  herr_t status;
  hid_t ememspace;
  hid_t hdf5_status;

  hsize_t edimens_3d[ N_DIMS ];
  hsize_t estart_3d[ N_DIMS ], ecount_3d[ N_DIMS ], estride_3d[ N_DIMS ], eblock_3d[ N_DIMS ];
  hsize_t dstart_3d[ N_DIMS ], dcount_3d[ N_DIMS ], dstride_3d[ N_DIMS ], dblock_3d[ N_DIMS ];

  double t_start = seconds();

  // Init Par HDF5
  H5Eset_current_stack (H5E_DEFAULT);

  plist_id = H5Pcreate (H5P_FILE_ACCESS);
  hdf5_status = H5Pset_fapl_mpio (plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  sprintf(fname,"density_t.%d.h5", step);
  file_id = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);

  edimens_3d[ 0 ] = NXP2;
  edimens_3d[ 1 ] = NYP2;
  edimens_3d[ 2 ] = NZP2;

  ememspace = H5Screate_simple( N_DIMS, edimens_3d, NULL );

  edimens_3d[ 0 ] = grid_global_dimensions[ 0 ];
  edimens_3d[ 1 ] = grid_global_dimensions[ 1 ];
  edimens_3d[ 2 ] = grid_global_dimensions[ 2 ];

  efilespace = H5Screate_simple( N_DIMS, edimens_3d, NULL );

  estart_3d[ 0 ] = 1;
  estart_3d[ 1 ] = 1;
  estart_3d[ 2 ] = 1;
  estride_3d[ 0 ] = 1;
  estride_3d[ 1 ] = 1;
  estride_3d[ 2 ] = 1;
  eblock_3d[ 0 ] = NX;
  eblock_3d[ 1 ] = NY;
  eblock_3d[ 2 ] = NZ;
  ecount_3d[ 0 ] = 1;
  ecount_3d[ 1 ] = 1;
  ecount_3d[ 2 ] = 1;

  dstart_3d[ 0 ] = LXS;
  dstart_3d[ 1 ] = LYS;
  dstart_3d[ 2 ] = LZS;
  dstride_3d[ 0 ] = 1;
  dstride_3d[ 1 ] = 1;
  dstride_3d[ 2 ] = 1;
  dblock_3d[ 0 ] = NX;
  dblock_3d[ 1 ] = NY;
  dblock_3d[ 2 ] = NZ;
  dcount_3d[ 0 ] = 1;
  dcount_3d[ 1 ] = 1;
  dcount_3d[ 2 ] = 1;

  status = H5Sselect_hyperslab( ememspace, H5S_SELECT_SET, estart_3d, estride_3d, ecount_3d, eblock_3d );
  status = H5Sselect_hyperslab( efilespace, H5S_SELECT_SET, dstart_3d, dstride_3d, dcount_3d, dblock_3d);

  dataset_id = H5Dopen (file_id, "/rho1", H5P_DEFAULT);

  double t_start_reading = seconds();
  status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, ememspace, efilespace, H5P_DEFAULT, rho);
  double t_end = seconds();

  if( mype == 0 ){

    fprintf( stdout, "\n\n\tStep = %d", step );
    fprintf( stdout, "\n\tTime of read_rho = %.3g (seconds).", t_end - t_start );
    fprintf( stdout, "\n\tTime of H5Dread = %.3g (seconds).", t_end - t_start_reading );
    fprintf( stdout, "\n\tMeasured bandwidth of H5Dread = %.3g (MB/s).", ( (double) sizeof(double) * grid_global_dimensions[ 0 ] * grid_global_dimensions[ 1 ] * grid_global_dimensions[ 2 ] ) / 1000000 / ( t_end - t_start_reading ) );    
  }

  status = H5Pclose( plist_id );
  status = H5Dclose(dataset_id);
  status = H5Fclose(file_id);
}


/* This program is created to benchmark parallel I/O based on HDF5 on CINECA Marconi */ 
int main (int argc, char * argv[])
{

  double *rho = NULL;
  int istep;

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &npes );
  MPI_Comm_rank( MPI_COMM_WORLD, &mype );

  if( argc < 3 && !mype ) { /* missing command line parameters */
    fprintf (stderr, "\nFatal Error: command line missing parameters! " );
    fprintf (stderr, "\nPlease run the command as ./prog.x NX NY NZ\n\n" );
    MPI_Abort( MPI_COMM_WORLD, 1 );
  }
  
  int grid_global_dimensions[3];
  grid_global_dimensions[0] = atoi( argv[1] );
  grid_global_dimensions[1] = atoi( argv[2] );
  grid_global_dimensions[2] = atoi( argv[3] );

  // Initialize MPI processes cartesia grid
  define_process_grid( grid_global_dimensions );

  NXP2 = NX + 2;
  NYP2 = NY + 2;
  NZP2 = NZ + 2;

  size_t n_rho_elements = NXP2 * NYP2 * NZP2;
  size_t byte_size_rho = n_rho_elements * sizeof( double );

  posix_memalign ( (void*) &rho, 4096, byte_size_rho );
  memset( rho, 0, byte_size_rho );

  if( mype == 0 ){

    fprintf( stdout, "\n\n\tStarting I/O Benchmark for HDF5 with %d processes distributed on a %d x %d x %d grid", npes, NPX, NPY, NPZ );
    fprintf( stdout, "\n\tDomain SIZE = { %d, %d, %d }, PEs local SIZE = { %d, %d, %d }", grid_global_dimensions[ 0 ], grid_global_dimensions[ 1 ], grid_global_dimensions[ 2 ], NX, NY, NZ );
    fprintf( stdout, "\n\tTotol bytes = %ld, Total bytes per process = %ld", ( (size_t) grid_global_dimensions[ 0 ] * grid_global_dimensions[ 1 ] * grid_global_dimensions[ 2 ] ) * sizeof( double ), byte_size_rho );
  }


  init_sphere( rho, grid_global_dimensions );
  
  for( istep = 0; istep < N_STEPS; istep++ ){
    
    pbc( rho );
    evolve_sphere( rho, grid_global_dimensions, istep );
    write_rho( istep, rho, grid_global_dimensions );
    //    memset( rho, 0, byte_size_rho );
    //    read_rho( istep, rho, grid_global_dimensions );
  }

  free( rho );

  MPI_Finalize();
    
}

