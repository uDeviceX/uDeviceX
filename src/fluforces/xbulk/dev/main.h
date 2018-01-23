__device__ void fetch(BCloud c, int i, forces::Pa *p) {
    float4 r, v;
    r = c.pp[2*i + 0];
    v = c.pp[2*i + 1];

    forces::r3v3k2p(r.x, r.y, r.z,
                    v.x, v.y, v.z,
                    SOLVENT_KIND, /**/ p);
    
    if (multi_solvent)
        p->color = c.cc[i];
}

__device__ int3 get_cid(const forces::Pa *pa) {
    int3 c;
    c.x = pa->x + XS/2;
    c.y = pa->y + YS/2;
    c.z = pa->z + ZS/2;
    return c;
}

__device__ void one_cell(int ia, forces::Pa pa, BCloud c, int start, int count, float rnd, /**/ Force *fa, Force *ff) {
    enum {X, Y, Z};
    int i, ib;
    forces::Pa pb;
    forces::Fo f;
    float *fb;
    
    for (i = 0; i < count; ++i) {
        ib = start + i;
        if (ib > ia) return;
        
        fetch(c, ib, &pb);
        fb = ff[ib].f;
        
        forces::force(pa, pb, rnd, /**/ &f);
        
        fa->f[X] += f.x;
        fa->f[Y] += f.y;
        fa->f[Z] += f.z;

        atomicAdd(fb + X, -f.x);
        atomicAdd(fb + Y, -f.y);
        atomicAdd(fb + Z, -f.z);
    }
}

__global__ void apply(int n, BCloud cloud, const int *start, const int *count, float rnd, /**/ Force *ff) {
    int ia;
    int3 cid;
    forces::Pa pa;

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(cloud, ia, &pa);
    cid = get_cid(&pa);

    
}
