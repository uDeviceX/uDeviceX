/*
   cudaError_t cudaMemcpy (void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)

*/

int nsol = particles->size, nrbc = rbcscoll->pcount(), szp = sizeof(Particle);
Particle *p_host = particles_datadump.data;
Particle *sol_dev = particles->xyzuvw.data, *rbc_dev = rbcscoll->xyzuvw.data;

printf("(simulation.hack.h) %d %d\n",nsol, nrbc);
  cudaMemcpy(p_host       , sol_dev      , szp*nsol, cudaMemcpyDeviceToHost);
if (rbcscoll)
  cudaMemcpy(p_host + nsol, rbc_dev      , szp*nrbc, cudaMemcpyDeviceToHost);

hello_a();

  cudaMemcpy(sol_dev      , p_host       , szp*nsol, cudaMemcpyHostToDevice);
if (rbcscoll)
  cudaMemcpy(rbc_dev      , p_host + nsol, szp*nrbc, cudaMemcpyHostToDevice);
