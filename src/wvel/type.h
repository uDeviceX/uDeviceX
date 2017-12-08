/* device related structures            */
/* (what is passed to device functions) */

struct WvelCste_d {
    float3 u;
};

struct WvelShear_d {
    float3 gdot; // shear rate in all three directions
};

enum {
    WALL_VEL_DEV_CSTE,
    WALL_VEL_DEV_SHEAR,
};

union WvelPar_d {
    WvelCste_d cste;
    WvelShear_d shear;
};

struct Wvel_d {
    WvelPar_d p;
    int type;
};

typedef void (*wvel_fun) (WvelPar_d, Coords, float3, /**/ float3*); 


/* data stored on the host                       */
/* (helpers to setup the above at each timestep) */

struct WvelCste {
    float3 u; // velocity amplitude
};

struct WvelShear {
    float3 g; // gdot amplitude
};

struct WvelShearSin {
    float3 g; // gdot amplitude
    float w;  // frequency
    int log_freq;
};

enum {
    WALL_VEL_CSTE,
    WALL_VEL_SHEAR,
    WALL_VEL_SHEAR_SIN,
};

union WvelPar {
    WvelCste cste;
    WvelShear shear;
    WvelShearSin shearsin;
};

/* main structure */

struct Wvel {
    Wvel_d dev; /* to be passed to device */
    WvelPar p;  /* parameters             */
    int type;
};
