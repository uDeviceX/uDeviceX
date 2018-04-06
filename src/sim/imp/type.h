/* solvent distribution */
struct FluDistr {
    DFluPack *p;
    DFluComm *c;
    DFluUnpack *u;
    DFluStatus *s;
};

/* particle exchanger for solvent forces */
struct FluExch {
    EFluPack *p;
    EFluComm *c;
    EFluUnpack *u;
};

/* rbc distribution */
struct RbcDistr {
    DRbcPack *p;
    DRbcComm *c;
    DRbcUnpack *u;
};

/* rigid distribution */
struct RigDistr {
    DRigPack *p;
    DRigComm *c;
    DRigUnpack *u;
};

/* object exchanger for forces */
struct ObjExch {
    EObjPack *p;
    EObjUnpack *u;
    EObjPackF *pf;
    EObjUnpackF *uf;
    EObjComm *c;
};

/* mesh exchanger */
struct Mexch {
    EMeshPack *p;
    EMeshComm *c;
    EMeshUnpack *u;
};

/* bounce back exchanger */
struct BBexch : Mexch {
    EMeshPackM *pm;
    EMeshCommM *cm;
    EMeshUnpackM *um;
};

struct Colorer {
    Mexch e;                 /* mesh exchanger     */
    Particle *pp;            /* particle workspace */
    float3 *minext, *maxext; /* bounding boxes     */
};

/* flux colorer */
struct Recolorer {
    bool flux_active;
    int flux_dir;
};

/* holder for bounce back tools and work */
struct BounceBack {
    MeshBB *d;
    Momentum *mm;
    BBexch e;
};

/* data holder for solvent */
struct Flu {
    FluQuants q;
    PairParams *params;

    FluDistr d;
    FluExch e;

    FluForcesBulk *bulk;
    FluForcesHalo *halo;

    Force *ff;
    Force *ff_hst; /* solvent forces on host    */

    float *ss;     /* stresses */
    float *ss_hst;

    float mass;  /* mass of one particle */ 
};

/* data holder for red blood cells */
struct Rbc {
    RbcQuants q;
    RbcDistr d;
    Force *ff;
    RbcForce *force;      /* helper to compute membrane forces */
    RbcParams *params;    /* model parameters */
    RbcCom    *com;     /* helper to compute center of masses */
    RbcStretch *stretch;  /* helper to apply stretching force to cells */
    MeshRead    *cell;     /* cell template */
    MeshWrite  *mesh_write;
    Triangles *tri; /* triangles for one cell on devices */
    float mass; /* mass of one particle */ 
};

/* data holder for rigid objects */
struct Rig {
    RigQuants q;
    Scan *ws; /* work for scan */
    Force *ff, *ff_hst;

    RigPinInfo *pininfo;
    RigDistr d;
    MeshWrite  *mesh_write;

    float mass;  /* mass of one particle */ 
};

/* data holder for walls */
struct Wall {
    Sdf *sdf;
    WallQuants q;
    WallTicket *t;
    Wvel *vel;
    WvelStep *velstep;
};

/* helper for computing object interactions */
struct ObjInter {
    ObjExch e;
    Contact *cnt;
    Fsi     *fsi;

    PairParams *cntparams;
    PairParams *fsiparams;
};

/* velocity controller */
struct Vcon {
    PidVCont *vcont;
    int log_freq;
    int adjust_freq;
    int sample_freq;
};

/* optional features */
struct Opt {
    bool fsi, cnt;
    bool flucolors, fluids, fluss;
    bool rbc, rbcids, rbcstretch;
    int rbcshifttype;
    bool rig, rig_bounce, rig_empty_pp;
    int rigshifttype;
    bool wall;
    bool inflow, outflow, denoutflow, vcon;
    bool dump_field, dump_parts, dump_strt, dump_rbc_com, dump_forces;
    float freq_field, freq_parts, freq_strt, freq_rbc_com;
    char strt_base_dump[FILENAME_MAX], strt_base_read[FILENAME_MAX];
    int recolor_freq;
    bool push_flu, push_rbc, push_rig;
};

struct Time {
    TimeLine *t;          /* current time manager         */
    float end, wall;      /* ent time, freeze time        */
    TimeStep *step;       /* time step manager            */
    TimeStepAccel *accel; /* helper for time step manager */
};

struct Dump {
    /* host particles for dump */
    Particle *pp;
    
    IoField *iofield;
    IoRig *iorig;
    IoBop *bop;
    DiagPart *diagpart; /* diagnostic */

    int id_bop, id_rbc, id_rbc_com, id_rig_mesh, id_strt;
};

struct Params {
    int3 L;          /* subdomain sizes */
    float kBT;       /* temperature     */
    int numdensity;  /* number density  */
};

struct Sim {
    /* quantities */
    Flu flu;
    Rbc rbc;
    Rig rig;
    Wall wall;

    Params params;
    BForce *bforce;

    /* helpers */
    Coords *coords;
    ObjInter objinter;
    BounceBack bb;
    Colorer colorer;
    Recolorer recolorer;
    Vcon vcon;
    Restrain *restrain;
    Dbg *dbg;
    MPI_Comm cart;

    /* open bc tools */
    Outflow *outflow;
    Inflow *inflow;
    DCont    *denoutflow;
    DContMap *mapoutflow;


    /* state */
    bool rigids;
    bool equilibrating;

    Opt opt;
    Dump dump;

    /* inter processing helpers */
    GenColor *gen_color;
};
