struct WvelCste {
    float3 u;
};

struct WvelShear {
    float3 gdot; // shear rate in all three directions
};

enum {
    WALL_VEL_CSTE,
    WALL_VEL_SHEAR,
};

union WvelPar {
    WvelCste cste;
    WvelShear shear;
};


struct Wvel {
    WvelPar p;
    int type;
};

typedef void (*wvel_fun) (WvelPar, Coords, float3, /**/ float3*); 
