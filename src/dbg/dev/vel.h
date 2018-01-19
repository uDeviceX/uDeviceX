static __device__ bool valid_vel(float v, int L, bool verbose) {
    float dx = fabs(v * dt);
    if (dx > L/2) {
        if (verbose) printf("DBG: v = %g (L = %d)\n", v, L);
        return false;
    }
    return true;
}

static __device__ bool valid_vel3(float vx, float vy, float vz, bool verbose) {
    bool ok = true;
    ok &= valid_vel(vx, XS, verbose);
    ok &= valid_vel(vy, YS, verbose);
    ok &= valid_vel(vz, ZS, verbose);

    return ok;
}

static __device__ err_type valid_vv(const Particle *p, bool verbose) {
    err_type e;
    const float *v = p->v;
    e = check_float3(v);
    if (e != ERR_NONE) return e;
    if ( valid_vel3(v[X], v[Y], v[Z], verbose)) e = ERR_NONE;
    else                                        e = ERR_INVALID;
    return e;
}

__global__ void check_vv(const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = valid_vv(pp + i, verbose);
    report(e);
}
