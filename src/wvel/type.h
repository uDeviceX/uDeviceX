/* device related structures            */
/* (what is passed to device functions) */

struct WvelCste_d {
    float3 u;
};

struct WvelShear_d {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // [0,1] : 1 if only lower wall is moving
};

struct WvelHS_d {
    float u; // radial max inflow
    float h; // height of the channel
};

enum {
    WALL_VEL_DEV_CSTE,
    WALL_VEL_DEV_SHEAR,
    WALL_VEL_DEV_HS,
};

union WvelPar_d {
    WvelCste_d cste;
    WvelShear_d shear;
    WvelHS_d hs;
    
};

/* device structure: to be passed to device code */

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
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // [0,1] : 1 if only lower wall is moving
};

struct WvelShearSin {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // [0,1] : 1 if only lower wall is moving
    float w;        // frequency
    int log_freq;
};

struct WvelHS {
    float u; // radial max inflow
    float h; // height of the channel    
};

enum {
    WALL_VEL_CSTE,
    WALL_VEL_SHEAR,
    WALL_VEL_SHEAR_SIN,
    WALL_VEL_HS,
};

union WvelPar {
    WvelCste cste;
    WvelShear shear;
    WvelShearSin shearsin;
    WvelHS hs;
};

/* main structure */

struct Wvel {
    // Wvel_d dev; /* to be passed to device */
    WvelPar p;  /* parameters             */
    int type;
};
