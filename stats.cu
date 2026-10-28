#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

#include "types.cuh"
#include "device.cuh"

void error_signal(const char *file, int line, const char *fmt, ...);
void error_report();
void error_abort();
void msg_print(const char *fmt, ...);

#define ERR(fmt, ...) do { \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report(); \
        error_abort(); \
    } while (0)

static __device__ float sumsq(Force f) {
    enum {X, Y, Z};
    float x, y, z;
    x = f.f[X]; y = f.f[Y]; z = f.f[Z];
    return x*x + y*y + z*z;
}

struct SumSq {
    __device__ float operator()(Force f) const { return sumsq(f); }
};

float force_stat_max(int n, const Force *dev) {
    using namespace thrust;
    if (!d::is_device_pointer(dev))
        ERR("`dev` is not a device pointer");
    float init, m;
    SumSq           unary;
    maximum<float> binary;
    device_ptr<const Force> beg(dev), end(dev+n);
    init = 0;
    try {
        m = transform_reduce(beg, end, unary, init, binary);
    } catch (system_error e) {
        msg_print("thrust::transform_reduce failed");
        ERR("%s", e.what());
    }
    return sqrt(m);
}
