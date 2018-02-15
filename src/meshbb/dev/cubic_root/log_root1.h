struct CubicInfo {
    real a, b, c, d;
    real h; /* root */
    bool status;
};

#define MAX_CUBIC_INFO 1000
static __device__ int ncubicInfo;
static __device__ CubicInfo cubicInfo[MAX_CUBIC_INFO];

static __device__ void log_cubic(real a, real b, real c, real d, real h, bool status) {
    CubicInfo i;
    i.a = a; i.b = b; i.c = c; i.d = d; i.h = h; i.status = status;
    cubicInfo[ncubicInfo++] = i;
    assert(ncubicInfo <= MAX_CUBIC_INFO);
}

static __device__ bool cubic_root(real a, real b, real c, real d, /**/ real *h) {
    bool rc;
    rc = cubic_root0(a, b, c, d, /**/ h);
    log_cubic(a, b, c, d, *h, rc);
    return rc;
}
