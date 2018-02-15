template <typename Farray>
struct Fo {
    Farray farray;
    int i;
};

static __device__ float random(int aid, int bid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? aid : bid;
    a2 = mask ? bid : aid;
    return rnd::mean0var1uu(seed, a1, a2);
}

template<typename Par, typename Parray, typename PFo>
static __device__ void force0(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> bfrag, const Map m, const PairPa a, int aid,
                              /**/ PFo *fa) {
    PairPa b;
    PFo f;
    int i;
    int bid; /* id of b */
    float rndval;
    
    for (i = 0; !endp(m, i); i ++ ) {
        bid = m2id(m, i);
        parray_get(bfrag.parray, bid, /**/ &b);
        rndval = random(aid, bid, rnd.seed, rnd.mask);
        pair_force(params, a, b, rndval, /**/ &f);
        pair_add(&f, /**/ fa);
    }
}

template<typename Par, typename Parray, typename Farray>
static __device__ void force1(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> frag, const Map m, const PairPa p, int id, Fo<Farray> f) {
    auto fa = farray_fo0(f.farray);
    force0(params, rnd, frag, m, p, id, /**/ &fa);
    farray_atomic_add<1>(&fa, f.i, f.farray);
}

template<typename Par, typename Parray, typename Farray>
static __device__ void force2(Par params, int3 L, const RFrag_v<Parray> frag, const flu::RndFrag rnd, PairPa p, int id, /**/ Fo<Farray> f) {
    Map m;    
    m = r2map(L, frag, p.x, p.y, p.z);
    p.x -= frag.dx * L.x;
    p.y -= frag.dy * L.y;
    p.z -= frag.dz * L.z;
    
    force1(params, rnd, frag, m, p, id, /**/ f);
}

template <typename Farray>
static __device__ Fo<Farray> i2f(const int *ii, Farray farray, int i) {
    /* local id and index to force */
    Fo<Farray> f;
    f.i = ii[i];
    f.farray = farray;
    return f;
}

template <typename Parray, typename Farray>
static __device__ Fo<Farray> Lfrag2f(const LFrag_v<Parray> frag, Farray farray, int i) {
    return i2f(frag.ii, farray, i);
}

template <typename Par, typename Parray, typename Farray>
static __device__ void force3(Par params, int3 L, const LFrag_v<Parray> afrag, const RFrag_v<Parray> bfrag, const flu::RndFrag rnd, int i,
                              /**/ Farray farray) {
    PairPa p;
    Fo<Farray> f;
    parray_get(afrag.parray, i, &p);
    f = Lfrag2f(afrag, farray, i);
    force2(params, L, bfrag, rnd, p, i, f);
}

template <typename Par, typename Parray, typename Farray>
__global__ void apply(Par params, int3 L, const int27 start, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd,
                      /**/ Farray farray) {
    flu::RndFrag  rnd;
    RFrag_v<Parray> rfrag;
    LFrag_v<Parray> lfrag;
    int gid;
    int fid; /* fragment id */
    int i; /* particle id */

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= start.d[26]) return;
    fid = frag_dev::frag_get_fid(start.d, gid);
    i = gid - start.d[fid];
    lfrag = lfrags.d[fid];
    if (i >= lfrag.n) return;

    rfrag = rfrags.d[fid];
    assert_frag(L, fid, rfrag);

    rnd = rrnd.d[fid];
    force3(params, L, lfrag, rfrag, rnd, i, /**/ farray);
}
