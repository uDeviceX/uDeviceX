#define _S_ static __device__

_S_ bool cutoff_range(PairPa pa, PairPa pb) {
    float x, y, z;
    x = pa.x - pb.x;
    y = pa.y - pb.y;
    z = pa.z - pb.z;
    return x*x + y*y + z*z <= 1.f;
}

_S_ int3 get_cid(int3 L, const PairPa *pa) {
    int3 c;
    c.x = pa->x + L.x / 2;
    c.y = pa->y + L.y / 2;
    c.z = pa->z + L.z / 2;
    return c;
}

_S_ bool valid_c(int c, int hi) {
    return (c >= 0) && (c < hi);
}

_S_ bool valid_cid(int3 L, int3 c) {
    return
        valid_c(c.x, L.x) &&
        valid_c(c.y, L.y) &&
        valid_c(c.z, L.z);    
}

template <typename Par, typename Parray, typename Farray, typename Fo>
_S_ void loop_pp(const Par *params, int ia, PairPa pa, Parray parray, int start, int end, float seed, /**/ Fo *fa, Farray farray) {
    enum {X, Y, Z};
    int ib;
    PairPa pb;
    Fo f;
    float rnd;
    
    for (ib = start; ib < end; ++ib) {
        if (ib >= ia) continue;
        
        fetch(parray, ib, &pb);

        if (!cutoff_range(pa, pb)) continue;
        
        rnd = rnd::mean0var1ii(seed, ia, ib);
        pair_force(params, pa, pb, rnd, /**/ &f);

        pair_add(&f, /**/ fa);

        farray_atomic_add<-1>(&f, ib, /**/ farray);
    }
}

template <typename Par, typename PArray, typename Farray, typename Fo>
_S_ void one_row(const Par *params, int3 L, int dz, int dy, int ia, int3 ca, PairPa pa, PArray parray, const int *start, float seed,
                               /**/ Fo *fa, Farray farray) {
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

    loop_pp(params, ia, pa, parray, bs, be, seed, /**/ fa, farray);
}

// unroll loop
template <typename Par, typename Parray, typename Farray>
__global__ void apply(Par params, int3 L, int n, Parray parray, const int *start, float seed, /**/ Farray farray) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    PairPa pa;
    auto fa = farray_fo0(farray);

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(parray, ia, &pa);
    ca = get_cid(L, &pa);

#define ONE_ROW(dz, dy) one_row (&params, L, dz, dy, ia, ca, pa, parray, start, seed, /**/ &fa, farray)
    
    ONE_ROW(-1, -1);
    ONE_ROW(-1,  0);
    ONE_ROW(-1,  1);
    ONE_ROW( 0, -1);
    ONE_ROW( 0,  0);

#undef ONE_ROW

    farray_atomic_add<1>(&fa, ia, farray);
}

#undef _S_
