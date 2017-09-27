static __device__ float3 dpd0(uint aid, uint bid, float rnd) {
    float fx, fy, fz;
    forces::Pa a, b;
    forces::Fo f;

    cloud_get(aid, &a);
    cloud_get(bid, &b);
    forces::f32f(&fx, &fy, &fz, /**/ &f);
    forces::genf(a, b, rnd, /**/ f);
    return make_float3(fx, fy, fz);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ float3 dpd(uint aid, uint bid) {
    float rnd;
    rnd = random(aid, bid); /* (sic) */
    return dpd0(aid, bid, rnd);
}
