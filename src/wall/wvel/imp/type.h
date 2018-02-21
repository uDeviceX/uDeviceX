/* data stored on the host                       */
/* (helpers to setup the views at each timestep) */

struct WvelCste {
    float3 u; // velocity amplitude
};

struct WvelShear {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
};

struct WvelShearSin {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    float w;        // frequency
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
    WvelPar p;  /* parameters             */
    int type;
};

/* step */

union WvelPar_v {
    WvelCste_v cste;
    WvelShear_v shear;
    WvelHS_v hs;
};

/* device structure: to be passed to device code */

struct WvelStep {
    WvelPar_v p;
    int type;
};

