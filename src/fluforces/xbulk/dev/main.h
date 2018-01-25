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

__device__ void fetch(TBCloud c, int i, forces::Pa *p) {
    float4 r, v;
    r = fetch(c.pp, 2*i + 0);
    v = fetch(c.pp, 2*i + 1);

    forces::r3v3k2p(r.x, r.y, r.z,
                    v.x, v.y, v.z,
                    SOLVENT_KIND, /**/ p);
    
    if (multi_solvent)
        p->color = fetch(c.cc, i);
}

__device__ bool cutoff_range(forces::Pa pa, forces::Pa pb) {
    float x, y, z;
    x = pa.x - pb.x;
    y = pa.y - pb.y;
    z = pa.z - pb.z;
    return x*x + y*y + z*z <= 1.f;
}

__device__ int3 get_cid(const forces::Pa *pa) {
    int3 c;
    c.x = pa->x + XS/2;
    c.y = pa->y + YS/2;
    c.z = pa->z + ZS/2;
    return c;
}

__device__ bool valid_c(int c, int hi) {
    return (c >= 0) && (c < hi);
}

__device__ bool valid_cid(int3 c) {
    return
        valid_c(c.x, XS) &&
        valid_c(c.y, YS) &&
        valid_c(c.z, ZS);    
}

template<typename CL>
__device__ void one_cell(int ia, forces::Pa pa, CL c, int start, int end, float seed, /**/ float fa[3], Force *ff) {
    enum {X, Y, Z};
    int ib;
    forces::Pa pb;
    forces::Fo f;
    float *fb, rnd;
    
    for (ib = start; ib < end; ++ib) {
        if (ib >= ia) continue;
        
        fetch(c, ib, &pb);

        if (!cutoff_range(pa, pb)) continue;
        
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

__global__ void apply_simplest(int n, BCloud cloud, const int *start, float seed, /**/ Force *ff) {
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
                
                one_cell(ia, pa, cloud, start[ib], start[ib + 1], seed, /**/ fa, ff);
            }        
        }        
    }

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}

__global__ void apply_smarter(int n, BCloud cloud, const int *start, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia, dy, dz;
    int enddy, enddx;
    int startx, endx;
    int bs, be, cid0;
    int3 ca, cb;
    forces::Pa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(cloud, ia, &pa);
    ca = get_cid(&pa);

    for (dz = -1; dz <= 0; ++dz) {
        cb.z = ca.z + dz;
        if (!valid_c(cb.z, ZS)) continue;
        
        enddy = dz ? 1 : 0;
            
        for (dy = -1; dy <= enddy; ++dy) {
            cb.y = ca.y + dy;
            if (!valid_c(cb.y, YS)) continue;

            enddx = (dz == 0 && dy == 0) ? 0 : 1;
            
            startx =    max(    0, ca.x - 1    );
            endx   = 1 + min(XS-1, ca.x + enddx);

            cid0 = XS * (cb.y + YS * cb.z);

            bs = start[cid0 + startx];
            be = start[cid0 + endx];

            one_cell(ia, pa, cloud, bs, be, seed, /**/ fa, ff);
        }        
    }

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}

template<typename CL>
__device__ void one_row(int dz, int dy, int ia, int3 ca, forces::Pa pa, CL cloud, const int *start, float seed, /**/ float fa[3], Force *ff) {
    int3 cb;
    int enddx, startx, endx, cid0, bs, be;
    cb.z = ca.z + dz;
    cb.y = ca.y + dy;
    if (!valid_c(cb.z, ZS)) return;
    if (!valid_c(cb.y, YS)) return;

    /* dx runs from -1 to enddx */
    enddx = (dz == 0 && dy == 0) ? 0 : 1;

    startx =     max(   0, ca.x - 1    );
    endx   = 1 + min(XS-1, ca.x + enddx);

    cid0 = XS * (cb.y + YS * cb.z);

    bs = start[cid0 + startx];
    be = start[cid0 + endx];

    one_cell(ia, pa, cloud, bs, be, seed, /**/ fa, ff);
}

// unroll loop
__global__ void apply_unroll(int n, BCloud cloud, const int *start, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    forces::Pa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(cloud, ia, &pa);
    ca = get_cid(&pa);

#define ONE_ROW(dz, dy) one_row (dz, dy, ia, ca, pa, cloud, start, seed, /**/ fa, ff)
    
    ONE_ROW(-1, -1);
    ONE_ROW(-1,  0);
    ONE_ROW(-1,  1);
    ONE_ROW( 0, -1);
    ONE_ROW( 0,  0);

#undef ONE_ROW

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}

// textures
__global__ void apply(int n, TBCloud cloud, const int *start, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    forces::Pa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(cloud, ia, &pa);
    ca = get_cid(&pa);

#define ONE_ROW(dz, dy) one_row (dz, dy, ia, ca, pa, cloud, start, seed, /**/ fa, ff)
    
    ONE_ROW(-1, -1);
    ONE_ROW(-1,  0);
    ONE_ROW(-1,  1);
    ONE_ROW( 0, -1);
    ONE_ROW( 0,  0);

#undef ONE_ROW

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}
