static __device__ void cloud_get(uint i, /**/ forces::Pa *p) { /* local fetch */
    /* i: particle index */
    float4 r, v;
    float r0[3], v0[3];

    r = fetchF4(i);
    v = fetchF4(xadd(i, 1u));
    f4tof3(r, r0); f4tof3(v, v0);
    forces::rvk2p(r0, v0, SOLVENT_KIND, /**/ p);
}

inline __device__ void cloud_get_color(uint i, /**/ forces::Pa *p) {
    p->color = fetchC(i);
}

static __device__ void cloud_pos(/* dummy c */ int i, /**/ float *x, float *y, float *z) {
    float4 r;
    r = fetchH4(i);
    *x = r.x; *y = r.y; *z = r.z;
}
