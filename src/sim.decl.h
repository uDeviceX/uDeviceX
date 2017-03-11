namespace sim {
  bool rbcs0;

  int s_n;
  Particle *s_pp; /* Solvent */
  Force    *s_ff;

  int r_n, r_nc, r_nt = RBCnt, r_nv = RBCnv;
  Particle *r_pp; /* RBC */
  Force    *r_ff;

  float4  *s_zip0; /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  Particle *s_pp0; /* Solvent (temporal buffer) */

  CellLists* cells;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host */
  Particle      r_pp_hst[MAX_PART_NUM]; /* RBC on host */
  Force         r_ff_hst[MAX_PART_NUM]; /* RBC force on host */
  
  Particle      sr_pp[MAX_PART_NUM];    /* solvent + RBC on host */


  float r_rr0[3*MAX_VERT_NUM];  /* initial positions */
  float r_mass = rbc_mass, r_Iinv[6]; /* mass, moment of inertia */
  float r_com[3], r_e0[3], r_e1[3], r_e2[3]; /* COM,  basis vectors of the body */
  float r_v[3], r_om[3];  /* linear velocity, angular velocity*/
  float r_f[3], r_to[3];  /* force, torque */

#ifdef GWRP  
float rbc_xx[MAX_PART_NUM], rbc_yy[MAX_PART_NUM], rbc_zz[MAX_PART_NUM];
float sol_xx[MAX_PART_NUM], sol_yy[MAX_PART_NUM], sol_zz[MAX_PART_NUM];
int   iotags[MAX_PART_NUM];
#endif  
  
}
