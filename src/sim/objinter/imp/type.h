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
    ObjExch *e;
    Contact *cnt;
    Fsi     *fsi;

    PairParams *cntparams;
    PairParams *fsiparams;
};

