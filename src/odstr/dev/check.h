namespace odstr { namespace sub { namespace dev {

/* TODO : use comma */

#define DBG
#ifdef DBG
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
#else
static __device__ void check_cel(float x, int i, int L) {}
static __device__ void check_vel(float v, int L) {}
#endif

}}} // namespace
