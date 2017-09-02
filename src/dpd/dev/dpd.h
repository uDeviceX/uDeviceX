static __device__ void lfetch(uint i, /**/ forces::Pa *a) { /* local fetch */
    /* i: particle index */
    float4 r, v;
    float r0[3], v0[3];

    r = fetchF4(i);
    v = fetchF4(xadd(i, 1u));
    f4tof3(r, r0); f4tof3(v, v0);
    forces::rvk2p(r0, v0, SOLVENT_TYPE, /**/ a);
}

static __device__ float3 dpd0(uint aid, uint bid, float rnd) {
    float fx, fy, fz;
    forces::Pa a, b;

    lfetch(aid, &a);
    lfetch(bid, &b);

    forces::gen(a, b, rnd, &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ float3 dpd(uint aid, uint bid, uint dpid, uint spid) {
    float rnd;
    rnd = random(spid, dpid); /* (sic) */
    return dpd0(aid, bid, rnd);
}
