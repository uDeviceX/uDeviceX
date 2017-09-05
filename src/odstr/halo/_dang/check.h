namespace dev {
static __device__ int  valid0(float r, int L) {
    float lo, hi;
    lo = -2*L;
    hi =  2*L;
    return r > lo && r < hi;
};

static __device__ int  valid(const float r[3]) {
    enum {X, Y, Z};
    return valid0(r[X], XS) && valid0(r[Y], YS) && valid0(r[Z], ZS);
};

static __device__ void report(const float r[3]) {
    enum {X, Y, Z};
    printf("wild particle: [%g %g %g]\n", r[X], r[Y], r[Z]);
}

static __device__ void check(const float r[3]) {
    if (valid(r)) return;
    report(r);
    assert(0);
};
}
