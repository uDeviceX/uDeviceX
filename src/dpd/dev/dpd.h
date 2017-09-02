static __device__ float3 dpd0(float4 ra, float4 udest, float4 rb, float4 usrc, float rnd) {
    float fx, fy, fz;

    forces::Pa a;
    float r1[3], v1[3];
    f4tof3(ra, r1); f4tof3(udest, v1);
    forces::rvk2p(r1, v1, SOLVENT_TYPE, /**/ &a);

    forces::Pa b;
    float r2[3], v2[3];
    f4tof3(rb, r2);  f4tof3(usrc, v2);
    forces::rvk2p(r2, v2, SOLVENT_TYPE, /**/ &b);

    forces::gen(a, b, rnd, &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}


static __device__ float3 dpd1(uint bid, uint aid, float rnd) {
    float4 xdest, xsrc, udest, usrc;
    xdest = fetchF4(bid);
    udest = fetchF4(xadd(bid, 1u));

    xsrc  = fetchF4(aid);
    usrc  = fetchF4(xadd(aid, 1u));

    return dpd0(xdest, udest, xsrc, usrc, rnd);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ float3 dpd(uint bid, uint aid, uint dpid, uint spid) {
    float rnd;
    rnd = random(spid, dpid);
    return dpd1(bid, aid, rnd);
}
