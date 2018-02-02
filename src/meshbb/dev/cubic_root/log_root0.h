namespace dev {
static __device__ bool cubic_root(real dt0, real a, real b, real c, real d, /**/ real *h) {
    return cubic_root0(dt0, a, b, c, d, /**/ h);
}
}
