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

enum {
    RBC_RND0,
    RBC_RND1,
};

struct Rnd0_v {};
struct Rnd1_v {
    int *anti; /* indices of anti edges */
    float *rr; /* random numbers (managed by RbcRnd) */
};

union RndInfo {
    Rnd0_v rnd0;
    Rnd1_v rnd1;
};

struct RbcForce {
    RbcRnd *rnd;

    Adj   *adj;
    Adj_v *adj_v;
    
    int stype;
    SFreeInfo sinfo;
    int rtype;
    RndInfo rinfo;
};
