namespace Sim {
  DeviceBuffer<Particle>     *s_pp; /* Solvent */
  DeviceBuffer<Acceleration> *s_aa;

  DeviceBuffer<Particle>     *s_pp0; /* Solvent (temporal buffer) */
  DeviceBuffer<Acceleration> *s_aa0;

  DeviceBuffer<Particle>     *r_pp; /* RBC */
  DeviceBuffer<Acceleration> *r_aa;

  DeviceBuffer<float4 > *xyzouvwo;
  DeviceBuffer<ushort4> *xyzo_half;

  CellLists* cells;

  bool wall_created = false;
  bool sim_is_done = false;

  MPI_Comm activecomm;

  size_t nsteps;
  float driving_acceleration = 0;

  pthread_t thread_datadump;
  pthread_mutex_t mutex_datadump;
  pthread_cond_t request_datadump, done_datadump;
  bool datadump_pending = false;
  int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
  bool async_thread_initialized;

  PinnedHostBuffer<Particle>      *particles_datadump;
  PinnedHostBuffer<Acceleration>  *accelerations_datadump;

  cudaEvent_t evdownloaded;

#define NPMAX 5000000 /* TODO: */
  float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
  float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
  int iotags[NPMAX];
}
