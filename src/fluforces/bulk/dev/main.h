static __device__ bool cutoff_range(PairPa pa, PairPa pb) {
    float x, y, z;
    x = pa.x - pb.x;
    y = pa.y - pb.y;
    z = pa.z - pb.z;
    return x*x + y*y + z*z <= 1.f;
}

static __device__ int3 get_cid(int3 L, const PairPa *pa) {
    int3 c;
    c.x = pa->x + L.x / 2;
    c.y = pa->y + L.y / 2;
    c.z = pa->z + L.z / 2;
    return c;
}

static __device__ bool valid_c(int c, int hi) {
    return (c >= 0) && (c < hi);
}

static __device__ bool valid_cid(int3 L, int3 c) {
    return
        valid_c(c.x, L.x) &&
        valid_c(c.y, L.y) &&
        valid_c(c.z, L.z);    
}

template<typename Par, typename Parray>
static __device__ void loop_pp(Par params, int ia, PairPa pa, Parray parray, int start, int end, float seed, /**/ float fa[3], Force *ff) {
    enum {X, Y, Z};
    int ib;
    PairPa pb;
    PairFo f;
    float *fb, rnd;
    
    for (ib = start; ib < end; ++ib) {
        if (ib >= ia) continue;
        
        fetch(parray, ib, &pb);

        if (!cutoff_range(pa, pb)) continue;
        
        fb = ff[ib].f;

        rnd = rnd::mean0var1ii(seed, ia, ib);
        pair_force(params, pa, pb, rnd, /**/ &f);
        
        fa[X] += f.x;
        fa[Y] += f.y;
        fa[Z] += f.z;

        atomicAdd(fb + X, -f.x);
        atomicAdd(fb + Y, -f.y);
        atomicAdd(fb + Z, -f.z);
    }
}

template<typename Par, typename Parray>
__global__ void apply_simplest(Par params, int3 L, int n, Parray parray, const int *start, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia, ib;
    int3 ca, cb;
    PairPa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(parray, ia, &pa);
    ca = get_cid(L, &pa);

    for (cb.z = ca.z - 1; cb.z <= ca.z + 1; ++cb.z) {
        for (cb.y = ca.y - 1; cb.y <= ca.y + 1; ++cb.y) {
            for (cb.x = ca.x - 1; cb.x <= ca.x + 1; ++cb.x) {
                if (!valid_cid(L, cb)) continue;
                ib = cb.x + L.x * (cb.y + L.y * cb.z);
                
                loop_pp(params, ia, pa, parray, start[ib], start[ib + 1], seed, /**/ fa, ff);
            }        
        }        
    }

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}

template<typename Par, typename Parray>
__global__ void apply_smarter(Par params, int3 L, int n, Parray parray, const int *start, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia, dy, dz;
    int enddy, enddx;
    int startx, endx;
    int bs, be, cid0;
    int3 ca, cb;
    PairPa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(parray, ia, &pa);
    ca = get_cid(L, &pa);

    for (dz = -1; dz <= 0; ++dz) {
        cb.z = ca.z + dz;
        if (!valid_c(cb.z, L.z)) continue;
        
        enddy = dz ? 1 : 0;
            
        for (dy = -1; dy <= enddy; ++dy) {
            cb.y = ca.y + dy;
            if (!valid_c(cb.y, L.y)) continue;

            enddx = (dz == 0 && dy == 0) ? 0 : 1;
            
            startx =    max(     0, ca.x - 1    );
            endx   = 1 + min(L.x-1, ca.x + enddx);

            cid0 = L.x * (cb.y + L.y * cb.z);

            bs = start[cid0 + startx];
            be = start[cid0 + endx];

            loop_pp(params, ia, pa, parray, bs, be, seed, /**/ fa, ff);
        }        
    }

    atomicAdd(ff[ia].f + X, fa[X]);
    atomicAdd(ff[ia].f + Y, fa[Y]);
    atomicAdd(ff[ia].f + Z, fa[Z]);
}

template<typename Par, typename PArray>
__device__ void one_row(Par params, int3 L, int dz, int dy, int ia, int3 ca, PairPa pa, PArray parray, const int *start, float seed, /**/ float fa[3], Force *ff) {
    int3 cb;
    int enddx, startx, endx, cid0, bs, be;
    cb.z = ca.z + dz;
    cb.y = ca.y + dy;
    if (!valid_c(cb.z, L.z)) return;
    if (!valid_c(cb.y, L.y)) return;

    /* dx runs from -1 to enddx */
    enddx = (dz == 0 && dy == 0) ? 0 : 1;

    startx =     max(    0, ca.x - 1    );
    endx   = 1 + min(L.x-1, ca.x + enddx);

    cid0 = L.x * (cb.y + L.y * cb.z);

    bs = start[cid0 + startx];
    be = start[cid0 + endx];

    loop_pp(params, ia, pa, parray, bs, be, seed, /**/ fa, ff);
}

// unroll loop
template<typename Par, typename Parray>
__global__ void apply(Par params, int3 L, int n, Parray parray, const int *start, float seed, /**/ Force *ff) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    PairPa pa;
    float fa[3] = {0};

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(parray, ia, &pa);
    ca = get_cid(L, &pa);

#define ONE_ROW(dz, dy) one_row (params, L, dz, dy, ia, ca, pa, parray, start, seed, /**/ fa, ff)
    
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
