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

/* flux colorer */
struct Recolorer {
    bool flux_active;
    int flux_dir;
};

/* Tracer DPD particle colorer */
struct Tracers {
    bool active;
    int freq; /* recolor frequency */
    float R; /* radius where tracers are placed */
    float iniP; /* probability to create tracer */
    float delP; /* probability to delete tracer */
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
    float mbr_bb;         /* when to activate membrane bounce */
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
    
    IoBop *bop;
    DiagPart *diagpart; /* diagnostic */
    Sampler field_sampler;

    int id_field, id_bop, id_strt;
};

struct Sim {
    /* quantities */
    Flu flu;
    Wall *wall;
    Objects *obj;

    BForce *bforce;

    /* helpers */
    Time time;
    Coords *coords;
    ObjInter *objinter;
    Recolorer recolorer;
    Tracers tracers;
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
    bool active_mbr_bb;

    Opt opt;
    Dump dump;

    /* inter processing helpers */
    GenColor *gen_color;
};
