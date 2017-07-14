namespace k_bipsbatch {
static __constant__ unsigned int start[27];

static __constant__ SFrag        ssfrag[26]; /* "send" fragment : TODO */
static __constant__ Frag          ffrag[26]; /* "remote" fragment */
static __constant__ Rnd            rrnd[26];

struct Pa { /* local particle */
    float x, y, z;
    float vx, vy, vz;
    uint id;
};

struct Fo { float *x, *y, *z; }; /* force */

static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }
static __device__ void p2rv2(const float2 *p, uint i,
                             float  *x, float  *y, float  *z,
                             float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    p += 3*i;
    s0 = __ldg(p++); s1 = __ldg(p++); s2 = __ldg(p++);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa frag2p(const Frag frag, uint i) {
    Pa p;
    p2rv2(frag.pp, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    p.id = i;
    return p;
}

static __device__ void pair(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* pair force ; l, r: local and remote */
    float3 r1 = make_float3( l.x,  l.y,  l.z), r2 = make_float3( r.x,  r.y,  r.z);
    float3 v1 = make_float3(l.vx, l.vy, l.vz), v2 = make_float3(r.vx, r.vy, r.vz);
    float3  f = force(SOLVENT_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, rnd);
    *fx = f.x; *fy = f.y; *fz = f.z;
}

static __device__ float random(uint lid, uint rid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? lid : rid;
    a2 = mask ? rid : lid;
    return l::rnd::d::mean0var1uu(seed, a1, a2);
}

static __device__ void force0(const Rnd rnd, const Frag frag, const Map m, const Pa l, /**/
                              float *fx, float *fy, float *fz) {
    /* l, r: local and remote particles */
    Pa r;
    uint i;
    uint lid, rid; /* ids */
    float seed;
    int mask;
    float x, y, z; /* pair force */
    lid = l.id;    
    mask = rnd.mask;
    seed = rnd.seed;

    *fx = *fy = *fz = 0;
    for (i = threadIdx.x & 1; !endp(m, i); i += 2) {
        rid = m2id(m, i);
        r = frag2p(frag, rid);
        pair(l, r, random(lid, rid, seed, mask), &x, &y, &z);
        *fx += x; *fy += y; *fz += z;
    }
}

static __device__ void force1(const Rnd rnd, const Frag frag, const Map m, const Pa p, Fo f) {
    float x, y, z; /* force */
    force0(rnd, frag, m, p, /**/ &x, &y, &z);
    atomicAdd(f.x, x);
    atomicAdd(f.y, y);
    atomicAdd(f.z, z);
}

static __device__ void force2(const Frag frag, const Rnd rnd, Pa p, /**/ Fo f) {
    int dx, dy, dz;
    Map m;
    m = p2map(frag, p.x, p.y, p.z);

    dx = frag.dx; dy = frag.dy; dz = frag.dz; /* TODO: where it should be? */
    p.x -= dx * XS;
    p.y -= dy * YS;
    p.z -= dz * ZS;
    force1(rnd, frag, m, p, f);
}

static __device__ Fo i2f(const int *ii, float *ff, uint i) {
    /* local id and index to force */
    Fo f;
    ff += 3*ii[i];
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

static __device__ void p2rv(const float *p, uint i,
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    p += 6*i;
     *x = *(p++);  *y = *(p++);  *z = *(p++);
    *vx = *(p++); *vy = *(p++); *vz = *(p++);
}

static __device__ Pa sfrag2p(const SFrag sfrag, uint i) {
    Pa p;
    p2rv(sfrag.pp,     i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    p.id = i;
    return p;
}

static __device__ Fo sfrag2f(const SFrag sfrag, float *ff, uint i) {
    return i2f(sfrag.ii, ff, i);
}

static __device__ void force3(const SFrag sfrag, const Frag frag, const Rnd rnd, uint i, /**/ float *ff) {
    Pa p;
    Fo f;
    p = sfrag2p(sfrag, i);
    f = sfrag2f(sfrag, ff, i);
    force2(frag, rnd, p, f);
}

static __device__ unsigned int get_hid(const unsigned int a[], const unsigned int i) {
    /* where is `i' in sorted a[27]? */
    unsigned int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

__global__ void force(/**/ float *ff) {
    Frag frag;
    Rnd  rnd;
    SFrag sfrag;
    int gid;
    uint hid; /* halo id */
    uint i; /* particle id */

    gid = (threadIdx.x + blockDim.x * blockIdx.x) >> 1;
    if (gid >= start[26]) return;
    hid = get_hid(start, gid);
    i = gid - start[hid];
    sfrag = ssfrag[hid];
    if (i >= sfrag.n) return;

    frag = ffrag[hid];
    rnd = rrnd[hid];
    force3(sfrag, frag, rnd, i, /**/ ff);
}
}
