namespace sim {
  bool rbcs0;

  int s_n;
  Particle *s_pp; /* Solvent */
  Force    *s_ff;

  int r_n;
  Particle *r_pp; /* RBC */
  Force    *r_ff;

  float4  *s_zip0; /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  Particle *s_pp0; /* Solvent (temporal buffer) */

  CellLists* cells;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host       */
  Force         s_ff_hst[MAX_PART_NUM]; /* solvent force on host */
  Particle      r_pp_hst[MAX_PART_NUM]; /* RBC on host           */
  Force         r_ff_hst[MAX_PART_NUM]; /* RBC force on host     */
  
  Particle      sr_pp[MAX_PART_NUM];    /* solvent + RBC on host */


  float r_rr0_hst[3*MAX_VERT_NUM];                 /* initial positions */
  float *r_rr0;
    
  Solid  solid_hst;
  Solid *solid_dev;
    
#ifdef GWRP  
float rbc_xx[MAX_PART_NUM], rbc_yy[MAX_PART_NUM], rbc_zz[MAX_PART_NUM];
float sol_xx[MAX_PART_NUM], sol_yy[MAX_PART_NUM], sol_zz[MAX_PART_NUM];
int   iotags[MAX_PART_NUM];
#endif  
  
}
