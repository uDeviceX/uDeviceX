static __device__ bool valid_pos(float x, int L, bool verbose) {
    if (x < -0.5*L || x > 0.5*L) {
        if (verbose) printf("DBG: x = %g (L = %d)\n", x, L);
        return false;
    }
    return true;
}

static __device__ bool valid_unpacked_pos(float  x, float  y, float  z, bool verbose) {
    bool ok = true;
    ok &= valid_pos(x, XS, verbose);
    ok &= valid_pos(y, YS, verbose);
    ok &= valid_pos(z, ZS, verbose);
    
    return ok;
}

static __device__ err_type valid_pos(const Particle *p, bool verbose) {
    float x, y, z;
    err_type e;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    e = check_float3(p->r);
    if (e != ERR_NONE) return e;
    if (valid_unpacked_pos(x, y, z, verbose)) e = ERR_NONE;
    else                                      e = ERR_INVALID;
    return e;
}

__global__ void check_pos(const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = valid_pos(pp + i, verbose);
    report(e);
}

static __device__ bool valid_unpacked_pos_pu(float x, float y, float z, bool verbose) {
    bool ok = true;
    ok &= valid_pos(x, XS + 3, verbose);
    ok &= valid_pos(y, YS + 3, verbose);
    ok &= valid_pos(z, ZS + 3, verbose);

    return ok;
}

static __device__ err_type valid_pos_pu(const Particle *p, bool verbose) {
    float x, y, z;
    err_type e;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    e = check_float3(p->r);
    if (e != ERR_NONE) return e;
    if (valid_unpacked_pos_pu(x, y, z, verbose)) e = ERR_NONE;
    else                                         e = ERR_INVALID;
    return e;
}

__global__ void check_pos_pu(const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = valid_pos_pu(pp + i, verbose);
    report(e);
}
