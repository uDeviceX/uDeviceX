static __device__ float3 dpd0(float4 ra, float4 va, float4 rb, float4 vb, float rnd) {
    float fx, fy, fz;

    forces::Pa a;
    float r1[3], v1[3];
    f4tof3(ra, r1); f4tof3(va, v1);
    forces::rvk2p(r1, v1, SOLVENT_TYPE, /**/ &a);

    forces::Pa b;
    float r2[3], v2[3];
    f4tof3(rb, r2);  f4tof3(vb, v2);
    forces::rvk2p(r2, v2, SOLVENT_TYPE, /**/ &b);

    forces::gen(a, b, rnd, &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}


static __device__ float3 dpd1(uint aid, uint bid, float rnd) {
    float4 ra, rb, va, vb;
    ra = fetchF4(aid);
    va = fetchF4(xadd(aid, 1u));

    rb  = fetchF4(bid);
    vb  = fetchF4(xadd(bid, 1u));

    return dpd0(ra, va, rb, vb, rnd);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ float3 dpd(uint bid, uint aid, uint dpid, uint spid) {
    float rnd;
    rnd = random(spid, dpid);
    return dpd1(bid, aid, rnd);
}
