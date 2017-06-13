int npp;        /* number of frozen pp    */ 
Particle *pp;   /* Solid frozen particles */
Force    *ff;
    
Particle   pp_hst[MAX_PART_NUM]; /* Solid pp on host          */
Force      ff_hst[MAX_PART_NUM]; /* Solid ff on host          */
  
Mesh m_hst; /* mesh of solid on host   */
Mesh m_dev; /* mesh of solid on device */

/* [t]riangle [c]ells [s]tarts / [c]ounts / [i]ds */
int *tcs_hst, *tcc_hst, *tci_hst; /* [t]riangle cell-lists on host   */
int *tcs_dev, *tcc_dev, *tci_dev; /* [t]riangle cell-lists on device */
        
float *bboxes_hst; /* [b]ounding [b]oxes of solid mesh on host   */
float *bboxes_dev; /* [b]ounding [b]oxes of solid mesh on device */
    
Particle *i_pp_hst, *i_pp_dev;       /* particles representing vertices of ALL meshes of solid [i]nterfaces */
Particle *i_pp_bb_hst, *i_pp_bb_dev; /* buffers for BB multi-nodes */ 
    
int ns;     /* number of solid objects       */
int nps;    /* number of particles per solid */
Solid *ss_hst;  /* solid infos on host           */
Solid *ss_dev;  /* solid infos on device         */

Solid *ss_bb_hst;  /* solid buffer for bounce back, host   */
Solid *ss_bb_dev;  /* solid buffer for bounce back, device */

/* buffers of solids for dump; this is needed because we dump the BB F and T separetely */
Solid *ss_dmphst, *ss_dmpbbhst;

float rr0_hst[3*MAX_PSOLID_NUM];      /* initial positions; same for all solids */
float *rr0;
