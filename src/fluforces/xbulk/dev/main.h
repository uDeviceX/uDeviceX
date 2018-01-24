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

__device__ bool valid_cid(int3 c) {
    return
        (c.x >= 0) && (c.x < XS) &&
        (c.y >= 0) && (c.y < YS) &&
        (c.z >= 0) && (c.z < ZS);    
}

__device__ void one_cell(int ia, forces::Pa pa, BCloud c, int start, int count, float seed, /**/ float fa[3], Force *ff) {
    enum {X, Y, Z};
    int i, ib;
    forces::Pa pb;
    forces::Fo f;
    float *fb, rnd;
    
    for (i = 0; i < count; ++i) {
        ib = start + i;
        if (ib > ia) return;
        
        fetch(c, ib, &pb);
        fb = ff[ib].f;

        rnd = rnd::mean0var1ii(seed, ia, ib);
        forces::force(pa, pb, rnd, /**/ &f);
        
        fa[X] += f.x;
        fa[Y] += f.y;
        fa[Z] += f.z;

        atomicAdd(fb + X, -f.x);
        atomicAdd(fb + Y, -f.y);
        atomicAdd(fb + Z, -f.z);
    }
}

__global__ void apply(int n, BCloud cloud, const int *start, const int *count, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia, ib;
    int3 ca, cb;
    forces::Pa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(cloud, ia, &pa);
    ca = get_cid(&pa);

    for (cb.z = ca.z - 1; cb.z <= ca.z + 1; ++cb.z) {
        for (cb.y = ca.y - 1; cb.y <= ca.y + 1; ++cb.y) {
            for (cb.x = ca.x - 1; cb.x <= ca.x + 1; ++cb.x) {
                if (!valid_cid(cb)) continue;
                ib = cb.x + XS * (cb.y + YS * cb.z);
                
                one_cell(ia, pa, cloud, start[ib], count[ib], seed, /**/ fa, ff);
            }        
        }        
    }

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}
