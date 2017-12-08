struct WvelShear {
    float3 gdot; // shear rate in all three directions
};

struct WvelParam {
    union {
        WvelShear shear;
    } p;
    int type;
};

typedef void (*wvel_fun) (WvelParam, Coords, float3, /**/ float3*); 
