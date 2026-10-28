#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"

#define ceiln(m, n) (((m) + (n) - 1)/(n))
#define k_cnf(n) ceiln((n), 128), 128

__global__ void convert_pp2rr_current_kernel(long n, const Particle *pp, Positioncp *rr) {
    enum {X, Y, Z};
    const float *s;
    float *d;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    d = rr[i].rc;
    s = pp[i].r;
    d[X] = s[X];
    d[Y] = s[Y];
    d[Z] = s[Z];
}

__global__ void convert_pp2rr_previous_kernel(long n, const Particle *pp, Positioncp *rr) {
    enum {X, Y, Z};
    const float *s;
    float *d;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    d = rr[i].rp;
    s = pp[i].r;
    d[X] = s[X];
    d[Y] = s[Y];
    d[Z] = s[Z];
}

__global__ void convert_pp2f4_pos_kernel(int n, const float *pp, float4 *zpp) {
    enum {X, Y, Z};
    int i;
    const float *r, *v;
    float x, y, z;
    float vx, vy, vz;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    r = &pp[6*i];
    v = &pp[6*i + 3];
    x = r[X]; y = r[Y]; z = r[Z];
    vx = v[X]; vy = v[Y]; vz = v[Z];
    zpp[2*i] = make_float4(x, y, z, 0);
    zpp[2*i + 1] = make_float4(vx, vy, vz, 0);
}

void convert_pp2rr_current(long n, const Particle *pp, Positioncp *rr) {
    KL(convert_pp2rr_current_kernel, (k_cnf(n)), (n, pp, rr));
}

void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr) {
    KL(convert_pp2rr_previous_kernel, (k_cnf(n)), (n, pp, rr));
}

void convert_pp2f4_pos(int n, const float *pp, float4 *zpp) {
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    KL(convert_pp2f4_pos_kernel, (k_cnf(n)), (n, pp, zpp));
}
