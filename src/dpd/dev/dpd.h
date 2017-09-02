static __device__ float3 dpd0(int dpid, float4 rdest, float4 udest, float4 rsrc, float4 usrc, int spid) {
    enum {X, Y, Z};
    float rnd, fx, fy, fz;
    float r1[3], r2[3], v1[3], v2[3];
    forces::Pa a, b;
    rnd = rnd::mean0var1ii( info.seed, xmin( spid, dpid ), xmax( spid, dpid ) );
    f4tof3(rdest, r1); f4tof3(rsrc, r2);
    f4tof3(udest, v1); f4tof3(usrc, v2);

    forces::rvk2p(r1, v1, SOLVENT_TYPE, /**/ &a);
    forces::rvk2p(r2, v2, SOLVENT_TYPE, /**/ &b);
    forces::gen(a, b, rnd, &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}

static __device__ float3 dpd(uint dentry, uint sentry, uint dpid, uint spid) {
    float4 xdest, xsrc, udest, usrc;
    float rnd;
    rnd = random(spid, dpid);

    xdest = fetchF4(dentry);
    xsrc  = fetchF4(sentry);
    udest = fetchF4(xadd(dentry, 1u));
    usrc  = fetchF4(xadd(sentry, 1u));
    return dpd0(dpid, xdest, udest, xsrc, usrc, spid );
}
