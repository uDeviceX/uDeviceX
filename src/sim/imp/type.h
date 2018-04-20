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
    RbcForce   *force;    /* helper to compute membrane forces         */
    RbcParams  *params;   /* model parameters                          */
    RbcCom     *com;      /* helper to compute center of masses        */
    RbcStretch *stretch;  /* helper to apply stretching force to cells */
    MeshRead   *cell;     /* cell template                             */
    Triangles  *tri;      /* triangles for one cell on devices         */
    float mass;           /* mass of one particle                      */
    MeshWrite  *mesh_write;
};

/* data holder for rigid objects */
struct Rig {
    RigQuants q;
    Force *ff, *ff_hst;

    RigPinInfo *pininfo;
    RigDistr d;
    MeshWrite  *mesh_write;

    float mass;  /* mass of one particle */ 
};

/* velocity controller */
struct Vcon {
    PidVCont *vcont;
    int log_freq;
    int adjust_freq;
    int sample_freq;
};

/* optional features */

struct Time {
    TimeLine *t;          /* current time manager         */
    float eq, end;        /* freeze time, end time        */
    TimeStep *step;       /* time step manager            */
    TimeStepAccel *accel; /* helper for time step manager */
};

struct Sampler {
    GridSampler *s;
    GridSampleData *d;
};

struct Dump {
    /* host particles for dump */
    Particle *pp;
    
    IoRig *iorig;
    IoBop *bop;
    DiagPart *diagpart; /* diagnostic */
    Sampler field_sampler;

    int id_field, id_bop, id_rbc, id_rbc_com, id_rig_mesh, id_strt;
};

struct Sim {
    /* quantities */
    Flu flu;
    Rbc rbc;
    Rig rig;
    Wall *wall;

    BForce *bforce;

    /* helpers */
    Time time;
    Coords *coords;
    ObjInter *objinter;
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
    bool equilibrating;

    Opt opt;
    Dump dump;

    /* inter processing helpers */
    GenColor *gen_color;
};
