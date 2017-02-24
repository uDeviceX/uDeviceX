namespace sim {
  StaticDeviceBuffer<Particle> *s_pp; /* Solvent */
  StaticDeviceBuffer<Force>    *s_ff;

  StaticDeviceBuffer<Particle> *s_pp0; /* Solvent (temporal buffer) */
  StaticDeviceBuffer<Force>    *s_ff0;

  StaticDeviceBuffer<Particle> *r_pp; /* RBC */
  StaticDeviceBuffer<Force>    *r_ff;

  StaticDeviceBuffer<float4 > *xyzouvwo;
  StaticDeviceBuffer<ushort4> *xyzo_half;

  CellLists* cells;

  bool wall_created = false;
  MPI_Comm activecomm;

  size_t nsteps;
  float driving_force = 0;
  MPI_Comm myactivecomm, mycartcomm;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
  StaticHostBuffer<Particle>      *particles_datadump;
  StaticHostBuffer<Force>         *forces_datadump;

  cudaEvent_t evdownloaded;

#define NPMAX 5000000 /* TODO: */
  float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
  float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
  int iotags[NPMAX];
}
