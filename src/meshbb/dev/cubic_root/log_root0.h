namespace dev {
static __device__ bool cubic_root(real dt, real a, real b, real c, real d, /**/ real *h) {
    return cubic_root0(dt, a, b, c, d, /**/ h);
}
}
