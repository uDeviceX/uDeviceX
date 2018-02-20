struct StressFree {
    float *ll; /* eq. spring lengths */
    float *aa; /* eq. triangle areas */
};

struct StressFul {
    float l0; /* eq. spring length */
    float a0; /* eq. triangle area */
};

union SFreeInfo {
    StressFul sful;
    StressFree sfree;
};

struct RbcForce {
    RbcRnd *rnd;
    
    SFreeInfo sinfo;
};
