/* object exchanger for forces */
struct ObjExch {
    EObjPack *p;
    EObjComm *c;
    EObjUnpack *u;

    EObjPackF *pf;
    EObjCommF *cf;
    EObjUnpackF *uf;    
};

/* helper for computing object interactions */
struct ObjInter {
    ObjExch *e;
    Contact *cnt;
    Fsi     *fsi;

    PairParams *cntparams;
};

