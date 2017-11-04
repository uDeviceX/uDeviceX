namespace dev {
struct CubicInfo {
    real a, b, c, d;
    real h; /* root */
    bool status;
};

static __device__ bool cubic_root(real a, real b, real c, real d, /**/ real *h) {
    return cubic_root0(a, b, c, d, /**/ h);
}
}
