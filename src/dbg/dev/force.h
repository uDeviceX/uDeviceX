static __device__ bool valid_acc(float a, int L, bool verbose) {
    float dx = fabs(a * dt * dt);
    if (dx > 0.5*L) {
        if (verbose) printf("DBG: a = %g (L = %d)\n", a, L);
        return false;
    }
    return true;
}

static __device__ bool valid_unpacked_f(int3 L, float fx, float fy, float fz, bool verbose) {
    bool ok = true;
    ok &= valid_acc(fx, L.x, verbose);
    ok &= valid_acc(fy, L.y, verbose);
    ok &= valid_acc(fz, L.z, verbose);

    return ok;
}

static __device__ err_type valid_f(int3 L, const Force *f, bool verbose) {
    enum {X, Y, Z};
    float fx, fy, fz;
    err_type e;
    fx = f->f[X]; fy = f->f[Y]; fz = f->f[Z];
    e = check_float3(f->f);
    if (e != ERR_NONE) return e;
    if (valid_unpacked_f(L, fx, fy, fz, verbose)) e = ERR_NONE;
    else                                       e = ERR_INVALID;
    return e;
}

__global__ void check_ff(int3 L, const Force *ff, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = valid_f(L, ff + i, verbose);
    report(e);
}
