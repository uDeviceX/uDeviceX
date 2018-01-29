static __device__ bool valid_pos(float x, int L) {
    if (x < -0.5*L || x > 0.5*L)
        return false;
    return true;
}

static __device__ bool valid_unpacked_pos(int3 L, float x, float y, float z) {
    bool ok = true;
    ok &= valid_pos(x, L.x);
    ok &= valid_pos(y, L.y);
    ok &= valid_pos(z, L.z);
    
    return ok;
}

static __device__ err_type valid_pos(int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    float x, y, z;
    err_type e;
    x  = p.r[X];  y = p.r[Y];  z = p.r[Z];
    e = check_float3(p.r);
    if (e != ERR_NONE) return e;
    if (valid_unpacked_pos(L, x, y, z)) e = ERR_NONE;
    else                                e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: pos pu: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void check_pos(int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = valid_pos(L, p, verbose);
    report(e);
}

static __device__ bool valid_unpacked_pos_pu(int3 L, float x, float y, float z) {
    bool ok = true;
    ok &= valid_pos(x, L.x + 3);
    ok &= valid_pos(y, L.y + 3);
    ok &= valid_pos(z, L.z + 3);

    return ok;
}

static __device__ err_type valid_pos_pu(int3 L, const Particle p, bool verbose) {
    enum {X, Y, Z};
    float x, y, z;
    err_type e;
    x  = p.r[X];  y = p.r[Y];  z = p.r[Z];
    e = check_float3(p.r);
    if (e != ERR_NONE) return e;
    if (valid_unpacked_pos_pu(L, x, y, z)) e = ERR_NONE;
    else                                   e = ERR_INVALID;

    if (verbose && e != ERR_NONE)
        printf("DBG: pos pu: p = [%g, %g %g], [%g, %g, %g]\n",
               p.r[X], p.r[Y], p.r[Z], p.v[X], p.v[Y], p.v[Z]);

    return e;
}

__global__ void check_pos_pu(int3 L, const Particle *pp, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    if (i >= n) return;
    p = pp[i];
    err_type e = valid_pos_pu(L, p, verbose);
    report(e);
}
