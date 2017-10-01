static __device__ void dpd0(uint aid, uint bid, float rnd, /**/ forces::Fo *f) {
    forces::Pa a, b;
    cloud_get(aid, &a);
    cloud_get(bid, &b);
    forces::gen(a, b, rnd, /**/ f);
}

static __device__ float random(uint i, uint j) {
    return rnd::mean0var1ii(info.seed, xmin(i, j), xmax(i, j));
}
static __device__ void dpd(uint aid, uint bid, /**/ forces::Fo *f) {
    float rnd;
    rnd = random(aid, bid);
    dpd0(aid, bid, rnd, /**/ f);
}
