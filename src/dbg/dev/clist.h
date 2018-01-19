static __device__ int3 cell_coords(int3 L, int cid) {
    int3 c;
    c.x = cid % L.x;
    c.z = cid / (L.y * L.x);
    c.y = (cid - L.y * L.x * c.z) / L.x;
    return c;
}

static __device__ bool at_coord(float x, int c, int L, bool verbose, const char *dir) {
    x += L/2;
    if ((x < c) || (x > c + 1)) {
        if (verbose) printf("Particle out of its cell : %s = %f, c = %d, L = %d\n", dir, x, c, L);
        return false;
    }
    return true;
}

static __device__ bool at_coords(const float r[3], int3 coords, int3 L, bool verbose) {
    bool ok = true;
    ok &= at_coord(r[X], coords.x, L.x, verbose, "x");
    ok &= at_coord(r[Y], coords.y, L.y, verbose, "y");
    ok &= at_coord(r[Z], coords.z, L.z, verbose, "z");
    return ok;
}

static __device__ err_type valid_particle(int3 L, int cid, const Particle *p, bool verbose) {
    err_type e;
    e = check_float3(p->r); if (e != ERR_NONE) return e;
    e = check_float3(p->v); if (e != ERR_NONE) return e;

    int3 coords = cell_coords(L, cid);
    bool valid = at_coords(p->r, coords, L, verbose);
    if (!valid) e = ERR_INVALID;
    
    return e;
}

static __device__ err_type valid_cell(int3 L, int cid, int s, int c, const Particle *pp, bool verbose) {
    err_type e = ERR_NONE;
    int i;
    Particle p;
    
    for (i = 0; i < c; ++i) {
        p = pp[s + i];
        e = valid_particle(L, cid, &p, verbose);
        if (e != ERR_NONE) return e;
    }
    return e;
}

__global__ void check_clist(int3 ncells, const int *ss, const int *cc, const Particle *pp, bool verbose = true) {
    int i, c, s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= ncells.x * ncells.y * ncells.z) return;

    s = ss[i];
    c = cc[i];
    
    err_type e = valid_cell(ncells, i, s, c, pp, verbose);
    report(e);
}
