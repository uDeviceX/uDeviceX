enum {
    TYPE_NONE,
    TYPE_PLATE,
    TYPE_CIRCLE
};

struct ParamsCircle {
    float Rsq;  /* radius squared  */
    float3 c;   /* center          */
};

/* plane is described as a*x + b*y + c*z + d = 0 */
struct ParamsPlate {
    float a, b, c, d;
};

union ParamsU {
    ParamsPlate plate;
    ParamsCircle circle;
};

struct Outflow {
    int *kk;        /* die or stay alive?      */
    int *ndead_dev; /* number of kills on dev  */
    int ndead;      /* number of kils          */
    
    ParamsU params; /* shape parameters */
    int type;       /* shape type       */
};
