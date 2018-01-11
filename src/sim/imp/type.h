/* types local for sim:: */

/* solvent distribution */
struct FluDistr {
    distr::flu::DFluPack p;
    distr::flu::DFluComm c;
    distr::flu::DFluUnpack u;
};

/* particle exchanger for solvent forces */
struct FluExch {
    exch::flu::Pack p;
    exch::flu::Comm c;
    exch::flu::Unpack u;
};

/* rbc distribution */
struct RbcDistr {
    DRbcPack p;
    DRbcComm c;
    DRbcUnpack u;
};

/* rigid distribution */
struct RigDistr {
    DRigPack p;
    DRigComm c;
    DRigUnpack u;
};

/* object exchanger for forces */
struct ObjExch {
    exch::obj::Pack p;
    exch::obj::Unpack u;
    exch::obj::PackF pf;
    exch::obj::UnpackF uf;
    exch::obj::Comm c;
};

/* mesh exchanger */
struct Mexch {
    exch::mesh::Pack p;
    exch::mesh::Comm c;
    exch::mesh::Unpack u;
};

/* bounce back exchanger */
struct BBexch : Mexch {
    exch::mesh::PackM pm;
    exch::mesh::CommM cm;
    exch::mesh::UnpackM um;
};

struct Colorer {
    Mexch e;                 /* mesh exchanger     */
    Particle *pp;            /* particle workspace */
    float3 *minext, *maxext; /* bounding boxes     */
};

/* holder for bounce back tools and work */
struct BounceBack {
    meshbb::BBdata d;
    Momentum *mm;
    BBexch e;
};

/* data holder for solvent */
struct Flu {
    flu::Quants q;

    FluDistr d;
    FluExch e;

    BulkData *bulkdata;
    HaloData *halodata;

    Force *ff;
    Force *ff_hst; /* solvent forces on host    */
};

/* data holder for red blood cells */
struct Rbc {
    rbc::Quants q;
    rbc::force::TicketT tt;

    RbcDistr d;

    Force *ff;

    rbc::com::Helper  com;      /* helper to compute center of masses */
    rbc::stretch::Fo *stretch;  /* helper to apply stretching [fo]rce to cells */
};

/* data holder for rigid objects */
struct Rig {
    rig::Quants q;
    scan::Work ws; /* work for scan */
    Force *ff, *ff_hst;

    RigDistr d;
};  

/* data holder for walls */
struct Wall {
    Sdf *sdf;
    wall::Quants q;
    wall::Ticket t;
    Wvel vel;
    Wvel_v vview;
};

/* helper for computing object interactions */
struct ObjInter {
    ObjExch e;
    Contact *cnt;
    Fsi     *fsi;
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
    bool inflow, outflow, denoutflow, vcon;
};

struct Sim {
    /* quantities */
    Flu flu;
    Rbc rbc;
    Rig rig;
    Wall wall;

    /* helpers */
    Coords coords;
    ObjInter objinter;
    BounceBack bb;
    Colorer colorer;
    Vcon vcon;
    MPI_Comm cart;

    /* open bc tools */
    Outflow *outflow;
    Inflow *inflow;
    DCont    *denoutflow;
    DContMap *mapoutflow;
    
    /* particles on host for dump */
    Particle *pp_dump;
    bop::Ticket dumpt;

    /* runtime config */
    Config *cfg;

    /* state */
    bool solids0;
    bool equilibrating;

    Opt opt;
};
