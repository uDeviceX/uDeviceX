namespace dev {
static __device__ int  valid(float r, int L) {
    float lo, hi;
    lo = -L/2 - 1;
    hi =  L/2 + 1;
    return r > hi || r < lo;
};
static __device__ void check(const float r[3]) {};
}
