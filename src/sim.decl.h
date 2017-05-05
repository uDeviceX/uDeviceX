namespace sim
{
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

    H5FieldDump *dump_field;

    Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host           */
    Force         s_ff_hst[MAX_PART_NUM]; /* solvent forces on host    */
    Particle      r_pp_hst[MAX_PART_NUM]; /* Solid pp on host          */
    Force         r_ff_hst[MAX_PART_NUM]; /* Solid ff on host          */
  
    Particle      sr_pp[MAX_PART_NUM];    /* solvent + solid pp on host */

    Mesh m_hst; /* mesh of solid on host   */
    Mesh m_dev; /* mesh of solid on device */
    
    int nsolid;     /* number of solid objects       */
    int npsolid;    /* number of particles per solid */
    Solid *ss_hst;  /* solid infos on host           */
    Solid *ss_dev;  /* solid infos on device         */

    Solid *ss_bbhst;  /* solid buffer for bounce back, host   */
    Solid *ss_bbdev;  /* solid buffer for bounce back, device */

    /* buffers of solids for dump; this is needed because we dump the BB F and T separetely */
    Solid *ss_dmphst, *ss_dmpbbhst;

    float r_rr0_hst[3*MAX_PSOLID_NUM];      /* initial positions; same for all solids */
    float *r_rr0;
}
