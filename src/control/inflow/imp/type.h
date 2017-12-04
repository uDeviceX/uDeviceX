enum Type {
    TYPE_NONE,
    TYPE_PLATE,
    TYPE_CIRCLE,
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
    circle::Params circle;
};

union VParamsU {
    plate::VParams plate;
    circle::VParams circle;
};

struct Inflow {
    Desc d;
    Type t;
    ParamsU p;
    VParamsU vp;
};
