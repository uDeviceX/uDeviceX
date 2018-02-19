/* view structures                      */
/* (what is passed to device functions) */

struct WvelCste_v {
    float3 u;
};

struct WvelShear_v {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
};

struct WvelHS_v {
    float u; // radial max inflow
    float h; // height of the channel
};

