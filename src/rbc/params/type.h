/* model parameters */
struct RbcParams {
    float gammaC; /*  */
    float gammaT; /*  */
    float kBT;    /* temperature */
    float kb;     /* bending */
    float phi;    /* spontaneous angle */
    float ks;     /* spring constant */
    float x0;     /* spring extension relative to max. extension */
    float mpow;   /* exponent in POW spring */
    float ka;     /* global area constant */
    float kd;     /* local area constant */
    float kv;     /* volume constant */
    float dt0;    /* timestep */
};

