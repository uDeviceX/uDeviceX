namespace sim {
  StaticDeviceBuffer<Particle>     *s_pp; /* Solvent */
  StaticDeviceBuffer<Acceleration> *s_aa;

  StaticDeviceBuffer<Particle>     *s_pp0; /* Solvent (temporal buffer) */
  StaticDeviceBuffer<Acceleration> *s_aa0;

  StaticDeviceBuffer<Particle>     *r_pp; /* RBC */
  StaticDeviceBuffer<Acceleration> *r_aa;

  DeviceBuffer<float4 > *xyzouvwo;
  DeviceBuffer<ushort4> *xyzo_half;

  CellLists* cells;

  bool wall_created = false;
  MPI_Comm activecomm;

  size_t nsteps;
  float driving_acceleration = 0;
  MPI_Comm myactivecomm, mycartcomm;

  H5PartDump *dump_part_solvent = NULL;
  H5PartDump *dump_part;
  H5FieldDump *dump_field;

  int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
  PinnedHostBuffer<Particle>      *particles_datadump;
  PinnedHostBuffer<Acceleration>  *accelerations_datadump;

  cudaEvent_t evdownloaded;

#define NPMAX 5000000 /* TODO: */
  float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
  float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
  int iotags[NPMAX];
}
