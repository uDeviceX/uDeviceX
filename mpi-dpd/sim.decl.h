namespace sim {
  StaticDeviceBuffer<Particle> *s_pp; /* Solvent */
  StaticDeviceBuffer<Force>    *s_ff;

  StaticDeviceBuffer<Particle> *s_pp0; /* Solvent (temporal buffer) */
  StaticDeviceBuffer<Force>    *s_ff0;

  StaticDeviceBuffer<Particle> *r_pp; /* RBC */
  StaticDeviceBuffer<Force>    *r_ff;

  float4  *s_zip0;                   /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  CellLists* cells;

  bool wall_created = false;
  MPI_Comm activecomm;

  size_t nsteps;
  float driving_force = 0;
  MPI_Comm myactivecomm, mycartcomm;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  /* solvent + RBC data on a host */
  Particle      sr_pp[MAX_PART_NUM];

/* TODO: */
  float rbc_xx[MAX_PART_NUM], rbc_yy[MAX_PART_NUM], rbc_zz[MAX_PART_NUM];
  float sol_xx[MAX_PART_NUM], sol_yy[MAX_PART_NUM], sol_zz[MAX_PART_NUM];
  int   iotags[MAX_PART_NUM];
}
