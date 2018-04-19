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


/* object exchanger for forces */
struct ObjExch {
    EObjPack *p;
    EObjUnpack *u;
    EObjPackF *pf;
    EObjUnpackF *uf;
    EObjComm *c;
};

/* helper for computing object interactions */
struct ObjInter {
    ObjExch e;
    Contact *cnt;
    Fsi     *fsi;

    PairParams *cntparams;
    PairParams *fsiparams;
};

/* data holder for cell membranes */
struct Mbr {
    RbcQuants q;
    MbrDistr d;
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

struct Dump {
    Particle *pp; /* workspace on host */
};

struct Objects {
    Mbr *mbr;
    Rig *rig;
    Dump *dump;
    Coords *coords;
    MPI_Comm cart;
};
