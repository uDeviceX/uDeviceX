static __device__ err_type valid_particle(int cid, const Particle *p, bool verbose) {
    err_type e;
    e = check_float3(p->r); if (e != err::NONE) return e;
    e = check_float3(p->v); if (e != err::NONE) return e;

    // TODO
    
    return e;
}

static __device__ err_type valid_cell(int cid, int s, int c, const Particle *pp, bool verbose) {
    err_type e = err::NONE;
    int i;
    Particle p;
    
    for (i = 0; i < c; ++i) {
        p = pp[s + i];
        e = valid_particle(cid, &p, verbose);
        if (e != err::NONE) return e;
    }
    return e;
}

__global__ void check_clist(int ncells, const int *ss, const int *cc, const Particle *pp, bool verbose) {
    int i, c, s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= ncells) return;

    s = ss[i];
    c = cc[i];
    
    err_type e = valid_cell(i, s, c, pp, verbose);
    report(e);
}
