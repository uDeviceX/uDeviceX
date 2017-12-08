
/* should be dependant on versions */
static __device__ void wvel(float3 r, /**/ float3 *v) {
    v->x = 0;
    v->y = 0;
    v->z = 0;
}

static __device__ void bounce_vel(float3 rw, /* io */ float3* v) {
    float3 vw;
    wvel(rw, /**/ &vw);
    v->x = 2 * vw.x - v->x;
    v->y = 2 * vw.y - v->y;
    v->z = 2 * vw.z - v->z;
}
