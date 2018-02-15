struct Fo { float *x, *y, *z; }; /* force */

template<typename Par>
static __device__ void pair(Par params, const PairPa a, const PairPa b, float rnd,
                            /**/ float *fx, float *fy, float *fz) {
    PairFo f;
    pair_force(params, a, b, rnd, /**/ &f);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

static __device__ float random(int aid, int bid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? aid : bid;
    a2 = mask ? bid : aid;
    return rnd::mean0var1uu(seed, a1, a2);
}

template<typename Par, typename Parray>
static __device__ void force0(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> bfrag, const Map m, const PairPa a, int aid, /**/
                              float *fx, float *fy, float *fz) {
    PairPa b;
    int i;
    int bid; /* id of b */
    float x, y, z; /* pair force */

    *fx = *fy = *fz = 0;
    for (i = 0; !endp(m, i); i ++ ) {
        bid = m2id(m, i);
        parray_get(bfrag.parray, bid, /**/ &b);
        pair(params, a, b, random(aid, bid, rnd.seed, rnd.mask), &x, &y, &z);
        *fx += x; *fy += y; *fz += z;
    }
}

template<typename Par, typename Parray>
static __device__ void force1(Par params, const flu::RndFrag rnd, const RFrag_v<Parray> frag, const Map m, const PairPa p, int id, Fo f) {
    float x, y, z; /* force */
    force0(params, rnd, frag, m, p, id, /**/ &x, &y, &z);
    atomicAdd(f.x, x);
    atomicAdd(f.y, y);
    atomicAdd(f.z, z);
}

template<typename Par, typename Parray>
static __device__ void force2(Par params, int3 L, const RFrag_v<Parray> frag, const flu::RndFrag rnd, PairPa p, int id, /**/ Fo f) {
    Map m;    
    m = r2map(L, frag, p.x, p.y, p.z);
    p.x -= frag.dx * L.x;
    p.y -= frag.dy * L.y;
    p.z -= frag.dz * L.z;
    
    force1(params, rnd, frag, m, p, id, /**/ f);
}

static __device__ Fo i2f(const int *ii, float *ff, int i) {
    /* local id and index to force */
    Fo f;
    ff += 3*ii[i];
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

template <typename Parray>
static __device__ Fo Lfrag2f(const LFrag_v<Parray> frag, float *ff, int i) {
    return i2f(frag.ii, ff, i);
}

template<typename Par, typename Parray>
static __device__ void force3(Par params, int3 L, const LFrag_v<Parray> afrag, const RFrag_v<Parray> bfrag, const flu::RndFrag rnd, int i, /**/ float *ff) {
    PairPa p;
    Fo f;
    parray_get(afrag.parray, i, &p);
    f = Lfrag2f(afrag, ff, i);
    force2(params, L, bfrag, rnd, p, i, f);
}

template<typename Par, typename Parray>
__global__ void apply(Par params, int3 L, const int27 start, const LFrag_v26<Parray> lfrags, const RFrag_v26<Parray> rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
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
    force3(params, L, lfrag, rfrag, rnd, i, /**/ ff);
}
