namespace dev {

static __device__ int3 get_cidx(int3 L, const float r[3]) {
    enum {X, Y, Z};
    int3 c;
    c.x = floor((double) r[X] + L.x/2);
    c.y = floor((double) r[Y] + L.y/2);
    c.z = floor((double) r[Z] + L.z/2);

    c.x = min(L.x, max(0, c.x));
    c.y = min(L.y, max(0, c.y));
    c.z = min(L.z, max(0, c.z));
    return c;
}

static __device__ int min3(int a, int b, int c) {return min(a, min(b, c));}
static __device__ int max3(int a, int b, int c) {return max(a, max(b, c));}

static __device__ int3 min3(int3 a, int3 b, int3 c) {
    return make_int3(min3(a.x, b.x, c.x),
                     min3(a.y, b.y, c.y),
                     min3(a.z, b.z, c.z));
}

static __device__ int3 max3(int3 a, int3 b, int3 c) {
    return make_int3(max3(a.x, b.x, c.x),
                     max3(a.y, b.y, c.y),
                     max3(a.z, b.z, c.z));
}

static __device__ void get_cells(int3 L, const Particle *A, const Particle *B, const Particle *C,
                          /**/ int3 *lo, int3 *hi) {
    int3 ca, cb, cc;
    ca = get_cidx(L, A->r);
    cb = get_cidx(L, B->r);
    cc = get_cidx(L, C->r);

    *lo = min3(ca, cb, cc);
    *hi = max3(ca, cb, cc);
}

static __device__ void find_collisions_cell(int tid, int start, int count, const Particle *pp,
                                            const Particle *A, const Particle *B, const Particle *C,
                                            /**/ int *ncol, float4 *datacol, int *idcol) {
    int i, entry;
    Particle p;
    BBState state;
    float tc, u, v;
    
    for (i = start; i < start + count; ++i) {
        p = pp[i];

        state = intersect_triangle(A->r, B->r, C->r, A->v, B->v, C->v, &p, /*io*/ &tc, /**/ &u, &v);

        if (state == BB_SUCCESS) {
            entry = atomicAdd(ncol + i, 1);
            entry += i * MAX_COL;
            datacol[entry] = make_float4(tc, u, v, 0);
            idcol[entry] = tid;
        }            
    }
}

__global__ void find_collisions(int nm, int nt, int nv, const int4 *tt, const Particle *i_pp,
                                int3 L, const int *starts, const int *counts, const Particle *pp,
                                /**/ int *ncol, float4 *datacol, int *idcol) {
    
    int gid, mid, tid, ix, iy, iz, cid;
    Particle A, B, C;
    int4 tri;
    int3 str, end;
    gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= nm * nt) return;

    mid = gid / nt;
    tid = gid % nt;

    tri = tt[tid];

    A = i_pp[mid * nt + tri.x];
    B = i_pp[mid * nt + tri.y];
    C = i_pp[mid * nt + tri.z];

    get_cells(L, &A, &B, &C, /**/ &str, &end);
    
    for (iz = str.z; iz < end.z; ++iz) {
        for (iy = str.y; iy < end.y; ++iy) {
            for (ix = str.x; ix < end.x; ++ix) {
                cid = ix + L.x * (iy + L.y * iz);
                
                find_collisions_cell(tid, starts[cid], counts[cid], pp, &A, &B, &C, /**/ ncol, datacol, idcol);
            }
        }
    }
}

__global__ void select_collisions(int n, /**/ int *ncol, float4 *datacol, int *idcol) {
    int i, c, j, dst, src, argmin;
    float tmin;
    float4 d;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n) return;

    c = ncol[i];

    argmin = 0; tmin = 2*dt;
    for (j = 0; j < c; ++j) {
        src = MAX_COL * i + j;
        d = datacol[src];
        if (d.x < tmin) {
            argmin = j;
            tmin = d.x;
        }
    }

    if (c) {
        src = MAX_COL * i + argmin;
        dst = MAX_COL * i;
        idcol[dst]   = idcol[src];
        datacol[dst] = datacol[src];
        ncol[i] = 1; /* we can use find_collisions again */
    }
}

// __golbal__ void perform_collisions() {

// }

} // dev
