enum {
    TYPE_NONE,
    TYPE_PLANE,
    TYPE_CIRCLE
};

union ParamsU {
    plane::Params plane;
    circle::Params circle;
};

struct Outflow {
    int *kk;        /* die or stay alive?      */
    int *ndead_dev; /* number of kills on dev  */
    int ndead;      /* number of kils          */

    ParamsU params; /* shape parameters */
    int type;       /* shape type       */
};
