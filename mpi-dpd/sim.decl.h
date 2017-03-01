namespace sim {
  int s_n;
  Particle *s_pp; /* Solvent */
  Force    *s_ff;

  int r_n, r_nc, r_nt = RBCnt, r_nv = RBCnv;
  Particle *r_pp; /* RBC */
  Force    *r_ff;
  int      triplets[MAX_FACE_NUM];
  float    *r_orig_xyzuvw, *r_host_av, *r_addfrc;

  float4  *s_zip0; /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  Particle *s_pp0; /* Solvent (temporal buffer) */
  Force    *s_ff0;

  CellLists* cells;

  bool wall_created = false;
  float driving_force = 0;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  Particle      sr_pp[MAX_PART_NUM]; /* solvent + RBC data on host */

/* TODO: */
  float rbc_xx[MAX_PART_NUM], rbc_yy[MAX_PART_NUM], rbc_zz[MAX_PART_NUM];
  float sol_xx[MAX_PART_NUM], sol_yy[MAX_PART_NUM], sol_zz[MAX_PART_NUM];
  int   iotags[MAX_PART_NUM];
}
