enum {
    RBC_STRESS_FUL,
    RBC_STRESS_FREE
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

    int stype;
    SFreeInfo sinfo;
};
