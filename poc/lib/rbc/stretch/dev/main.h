static __device__ void apply0(const float *f, /**/ float *fb) {
    enum {X, Y, Z};
    fb[X] += f[X]; fb[Y] += f[Y]; fb[Z] += f[Z];
}

static __device__ void apply1(int i, int nv, const float *f, Force *ff) {
    int k;
    k = i % nv;
    apply0(&f[3*k], /**/ ff[i].f);
}

__global__ void apply(int n, int nv, const float *f, Force *ff) {
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    apply1(i, nv, f, /**/ ff);
}
