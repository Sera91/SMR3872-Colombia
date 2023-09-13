from dask_mpi import initialize
from dask.distributed import Client
import dask.dataframe as dd
from distributed.scheduler import logger
import socket

if __name__ == "__main__":
   print('I am before client inizialization')

   n_tasks = int(os.getenv('SLURM_NTASKS'))
   mem = os.getenv('SLURM_MEM_PER_CPU')
   mem = str(int(mem))+'MB'

   initialize(memory_limit=mem)

   dask_client = Client()

   dask_client.wait_for_workers(n_workers=(n_tasks-2))
    #dask_client.restart()

   num_workers = len(dask_client.scheduler_info()['workers'])
   print("%d workers available and ready"%num_workers)
   
   dd.read_hdf('/leonardo_work/ICT23_SMR3872/datasets/TNG_data/LGalaxies/LGalaxies_001.hdf5', '/x')  

   dask_client.shutdown()


