struct WvelParam {
    float3 gdot; // shear rate in all three directions
};

static __device__ void wvel(WvelParam p, Coords c, float3 r, /**/ float3 *v) {
    float3 rc; // relative to center
    local2center(c, r, /**/ &rc);

    v->x = gdot.x * rc.x;
    v->y = gdot.y * rc.y;
    v->z = gdot.z * rc.z;
}
