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

  Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host           */
  Force         s_ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
  Particle      r_pp_hst[MAX_PART_NUM]; /* Solid pp on host          */
  Force         r_ff_hst[MAX_PART_NUM]; /* Solid ff on host          */
  
  Particle      sr_pp[MAX_PART_NUM];    /* solvent + solid pp on host */


  int nsolid;     /* number of solid objects       */
  int npsolid;    /* number of particles per solid */
  Solid *ss_hst;  /* solid infos on host           */
  Solid *ss_dev;  /* solid infos on device         */

  float r_rr0_hst[3*MAX_PSOLID_NUM];      /* initial positions; same for all solids */
  float *r_rr0;
}
