template <typename Par>
static __device__ void fetch_p(Par, Cloud c, int i, /**/ PairPa *p) {
    cloud_get_p(c, i, /**/ p);
}

static __device__ void fetch_p(PairDPDCM, Cloud c, int i, /**/ PairPa *p) {
    cloud_get(c, i, /**/ p);
}
