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

template <typename Par, typename Parray, typename Farray, typename Fo>
_S_ void loop_pp(const Par *params, int ia, PairPa pa, Parray parray, const Map *m, float seed, /**/ Fo *fa, Farray farray) {
    enum {X, Y, Z};
    int ib;
    PairPa pb;
    Fo f;
    float rnd;
    
    for (i = 0; !map_end(m, i); ++i) {
        ib = map_get_id(m, i);
        if (ib >= ia) continue;
        
        fetch(parray, ib, &pb);

        if (!cutoff_range(pa, pb)) continue;
        
        rnd = rnd::mean0var1ii(seed, ia, ib);
        pair_force(params, pa, pb, rnd, /**/ &f);

        pair_add(&f, /**/ fa);

        farray_atomic_add<-1>(&f, ib, /**/ farray);
    }
}

// unroll loop
template <typename Par, typename Parray, typename Farray>
__global__ void apply(Par params, int3 L, int n, Parray parray, const int *start, float seed, /**/ Farray farray) {
    enum {X, Y, Z};
    int ia;
    int3 ca;
    PairPa pa;
    Map map;
    auto fa = farray_fo0(farray);

    ia = threadIdx.x + blockIdx.x * blockDim.x;
    if (ia >= n) return;
    
    fetch(parray, ia, &pa);
    ca = get_cid(L, &pa);

    map_build(L, ca, start, &map);

    loop_pp(&params, ia, pa, parray, &map, seed, /**/ &fa, farray);

    farray_atomic_add<1>(&fa, ia, farray);
}

#undef _S_
