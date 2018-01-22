/* view structures                      */
/* (what is passed to device functions) */

struct WvelCste_v {
    float3 u;
};

struct WvelShear_v {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // [0,1] : 1 if only lower wall is moving
};

struct WvelHS_v {
    float u; // radial max inflow
    float h; // height of the channel
};

enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};

union WvelPar_v {
    WvelCste_v cste;
    WvelShear_v shear;
    WvelHS_v hs;
    
};

/* device structure: to be passed to device code */

struct Wvel_v {
    WvelPar_v p;
    int type;
};

typedef void (*wvel_fun) (WvelPar_v, Coords, float3, /**/ float3*); 
