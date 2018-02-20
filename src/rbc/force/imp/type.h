enum {
    RBC_SFUL,
    RBC_SFREE
};

struct StressFree_v {
    float *ll; /* eq. spring lengths */
    float *aa; /* eq. triangle areas */
};

struct StressFul_v {
    float l0; /* eq. spring length */
    float a0; /* eq. triangle area */
};

union SFreeInfo {
    StressFul_v sful;
    StressFree_v sfree;
};

struct RbcForce {
    RbcRnd *rnd;

    Adj   *adj;
    Adj_v *adj_v;
    
    int stype;
    SFreeInfo sinfo;
};
