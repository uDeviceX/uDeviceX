enum Type {
    TYPE_NONE,
    TYPE_PLATE,
};
    
struct Desc {
    curandState_t *rnds; /* random states on device         */
    float3 *uu;          /* target flux                     */
    float *cumflux;      /* cumulative flux                 */
    int *ndev;           /* number of particles on device   */
    int2 nc;             /* number of cells in 2 directions */
};

union ParamsU {
    plate::Params plate;
};

union VParamsU {
    plate::VParams plate;
};

struct Inflow {
    Desc d;
    Type t;
    ParamsU p;
    VParamsU vp;
};
