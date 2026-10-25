enum {
    OUTFLOW_TYPE_NONE,
    OUTFLOW_TYPE_PLATE,
    OUTFLOW_TYPE_CIRCLE
};

struct OutflowParamsCircle {
    float Rsq;  /* radius squared  */
    float3 c;   /* center          */
};

/* plane is described as a*x + b*y + c*z + d = 0 */
struct OutflowParamsPlate {
    float a, b, c, d;
};

union OutflowParamsU {
    OutflowParamsPlate plate;
    OutflowParamsCircle circle;
};

struct Outflow {
    int *kk;        /* die or stay alive?      */
    int *ndead_dev; /* number of kills on dev  */
    int ndead;      /* number of kils          */
    
    OutflowParamsU params; /* shape parameters */
    int type;       /* shape type       */
};
