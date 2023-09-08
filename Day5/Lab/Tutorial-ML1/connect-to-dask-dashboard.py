from dask_mpi import initialize
from dask.distributed import Client
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
   dask_client.shutdown()

client = Client()

host = client.run_on_scheduler(socket.gethostname)
port = client.scheduler_info()['services']['dashboard']
login_node_address = "supercomputer.university.edu" # Provide address/domain of login node

logger.info(f"ssh -N -L {port}:{host}:{port} {login_node_address}")
