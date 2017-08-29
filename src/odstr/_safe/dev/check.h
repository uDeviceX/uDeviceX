namespace odstr { namespace sub { namespace dev {
static __device__ void check_cel(float x, int i, int L) {
    if (i < 0 || i >= L) {
        printf("odstr: i = %d (L = %d) from x = %g\n", i, L, x);
        assert(0);
    }
}

static __device__ void check_vel(float v, int L) {
    float dx = fabs(v * dt);
    if (dx >= L / 2) {
        printf("odstr: vel: v = %g\n", v);
        assert(0);
    }
}

static const float eps = 1e-6f;

static __device__ void rescue(int L, float *x) {
    if (*x < -L/2) *x = -L/2;
    if (*x >= L/2) *x =  L/2 - eps;
}

}}} // namespace
