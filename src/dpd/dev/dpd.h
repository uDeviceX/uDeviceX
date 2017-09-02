static __device__ float3 dpd0(float4 rdest, float4 udest, float4 rsrc, float4 usrc, float rnd) {
    enum {X, Y, Z};
    float fx, fy, fz;
    float r1[3], r2[3], v1[3], v2[3];
    forces::Pa a, b;
    f4tof3(rdest, r1); f4tof3(rsrc, r2);
    f4tof3(udest, v1); f4tof3(usrc, v2);

    forces::rvk2p(r1, v1, SOLVENT_TYPE, /**/ &a);
    forces::rvk2p(r2, v2, SOLVENT_TYPE, /**/ &b);
    forces::gen(a, b, rnd, &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}


static __device__ float3 dpd1(uint dentry, uint sentry, float rnd) {
    float4 xdest, xsrc, udest, usrc;
    xdest = fetchF4(dentry);
    xsrc  = fetchF4(sentry);
    udest = fetchF4(xadd(dentry, 1u));
    usrc  = fetchF4(xadd(sentry, 1u));
    return dpd0(xdest, udest, xsrc, usrc, rnd);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ float3 dpd(uint dentry, uint sentry, uint dpid, uint spid) {
    float rnd;
    rnd = random(spid, dpid);
    return dpd1(dentry, sentry, rnd);
}
