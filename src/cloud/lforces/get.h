static __device__ void tex2rv(uint i, /**/ float r[3], float v[3]) {
    float4 r4, v4;
    r4 = fetchF4(i);
    v4 = fetchF4(xadd(i, 1u));
    f4tof3(r4, /**/ r); f4tof3(v4, /**/ v);
}

static __device__ void cloud_get(uint i, /**/ forces::Pa *p) { /* local fetch */
    /* i: particle index */
    float r[3], v[3];
    tex2rv(i, /**/ r, v);
    forces::rvk2p(r, v, SOLVENT_KIND, /**/ p);
    if (multi_solvent)
        p->color = fetchC(i / 2);
}

static __device__ void cloud_pos(/* dummy c */ int i, /**/ float *x, float *y, float *z) {
    float4 r;
    r = fetchH4(i);
    *x = r.x; *y = r.y; *z = r.z;
}
