namespace sim {
  int s_n;
  Particle *s_pp; /* Solvent */
  Force    *s_ff;

  int r_n, r_nc, r_nt = RBCnt, r_nv = RBCnv;
  Particle *r_pp; /* RBC */
  Force    *r_ff;
  int      r_faces[MAX_FACE_NUM];
  float    *r_host_av;

  float4  *s_zip0; /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  Particle *s_pp0; /* Solvent (temporal buffer) */

  CellLists* cells;

  bool wall_created = false;
  float driving_force = 0;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host */
  Particle      r_pp_hst[MAX_PART_NUM]; /* RBC on host */
  Force         r_ff_hst[MAX_PART_NUM]; /* RBC force on host */
  
  Particle      sr_pp[MAX_PART_NUM];    /* solvent + RBC on host */


  float r_v[3], r_com[3];  /* linear velocity, COM */
  float r_om[3], r_Iinv[6], r_to[3];  /* angular velocity, moment of inertia, torque */
  float r_rr0[3*MAX_VERT_NUM];  /* initial positions */
  float r_e0[3], r_e1[3], r_e2[3];  /* basis vectors of the body */

#ifdef GWRP  
float rbc_xx[MAX_PART_NUM], rbc_yy[MAX_PART_NUM], rbc_zz[MAX_PART_NUM];
float sol_xx[MAX_PART_NUM], sol_yy[MAX_PART_NUM], sol_zz[MAX_PART_NUM];
int   iotags[MAX_PART_NUM];
#endif  
  
}
