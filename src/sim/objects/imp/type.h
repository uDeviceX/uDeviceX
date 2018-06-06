/* membrane distribution */
struct MbrDistr {
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
struct MeshExch {
    EMeshPack *p;
    EMeshComm *c;
    EMeshUnpack *u;
};

/* mesh momentum exchanger */
struct MeshMomExch {
    EMeshPackM *p;
    EMeshCommM *c;
    EMeshUnpackM *u;
};

/* workspace for solvent coloring */
struct Colorer {
    Particle *pp_mesh; /* particle workspace */
    float3 *lo, *hi;   /* bounding boxes     */
};

/* workspace for mesh bounce back */
struct BounceBackData {
    Positioncp *rr_cp;
    MeshMomExch *e;
    Momentum *mm;
};

/* common parts for membranes and rigid objects */
struct Obj {
    char name[FILENAME_MAX];
    char ic_file[FILENAME_MAX];
    float mass;           /* mass of one particle                      */
    MeshRead   *mesh;     /* cell template                             */
    MeshWrite  *mesh_write;
    MeshExch   *mesh_exch;
    BounceBackData *bbdata;
    
    PairParams *fsi;
    PairParams *adhesion;
    WallRepulsePrm *wall_rep_prm;
};

/* data holder for cell membranes */
struct Mbr : Obj {
    RbcQuants q;
    MbrDistr d;
    Force *ff;            /* large time scale forces                   */
    Force *ff_fast;       /* small time scale forces                   */
    RbcForce   *force;    /* helper to compute membrane forces         */
    RbcParams  *params;   /* model parameters                          */
    RbcCom     *com;      /* helper to compute center of masses        */
    RbcStretch *stretch;  /* helper to apply stretching force to cells */
    Triangles  *tri;      /* triangles for one cell on devices         */

    Colorer  *colorer;
};

/* data holder for rigid objects */
struct Rig : Obj {
    RigQuants q;
    Force *ff;

    RigPinInfo *pininfo;
    RigDistr d;
};

struct Dump {
    Particle *pp; /* workspace on host */
    long id, id_diag;
};

struct Objects {
    int nmbr, nrig;
    Mbr **mbr;
    Rig **rig;
    Opt opt;
    Dump *dump;
    Coords *coords;
    MPI_Comm cart;
    MeshBB *bb;
    bool active; /* false before completely generated */
};
