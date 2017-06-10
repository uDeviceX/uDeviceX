namespace safety {

enum {X, Y, Z};

namespace {

__device__ void treat_nan(float *a) {if (!isfinite(*a)) *a = 0.f;}

template <int L>
__device__ void rbound(float *x) {
    treat_nan(x);

    *x = min((float)L, max(-(float)L, *x));
}

__device__ void vbound(float *v) {
    treat_nan(v + X); treat_nan(v + Y); treat_nan(v + Z);

    const float v2 = v[X]*v[X] + v[Y]*v[Y] + v[Z]*v[Z];
    if (v2 > XS/(2*dt)) {
        const float s = rsqrtf(v2);
        v[X] *= s;
        v[Y] *= s;
        v[Z] *= s;
    }
}

__global__ void bound_k(Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float *r = pp[i].r;
    float *v = pp[i].v;

    rbound<XS>(r + X); rbound<YS>(r + Y); rbound<ZS>(r + Z);
    vbound(v);
}
}

void bound(Particle *pp, int n) {
    if (n)
    bound_k <<<k_cnf(n)>>> (pp, n);
}

}
