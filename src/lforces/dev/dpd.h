static __device__ float3 dpd0(uint aid, uint bid, float rnd, /**/ forces::Fo f) {
    forces::Pa a, b;

    cloud_get(aid, &a);
    cloud_get(bid, &b);
    forces::genf(a, b, rnd, /**/ f);
    return make_float3(*f.x, *f.y, *f.z);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ float3 dpd(uint aid, uint bid, /**/ forces::Fo f) {
    float rnd;
    rnd = random(aid, bid); /* (sic) */
    return dpd0(aid, bid, rnd, /**/ f);
}
