namespace rbc { namespace rnd {
static __device__ float get(const D *d, int i) {
    assert(i < D->max);
    return D.r[i];
}} /* namespace */
