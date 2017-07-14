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
    p2rv2(frag.pp,     i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    p.id = i;
    return p;
}

static __device__ void force0(const Rnd rnd, const Frag frag, const Map m, Pa l, Fo f) {
    /* l, r: local and remote particles */
    Pa r;
    uint lid, rid; /* ids */
    float x, y, z;
    float vx, vy, vz;
    float *fx, *fy, *fz;

     x = l.x;   y = l.y;   z = l.z;
    vx = l.vx; vy = l.vy; vz = l.vz;
    fx = f.x; fy = f.y; fz = f.z;
    lid = l.id;
    
    int mask = rnd.mask;
    float seed = rnd.seed;
    float xforce = 0, yforce = 0, zforce = 0;
    for (uint i = threadIdx.x & 1; !endp(m, i); i += 2) {
        rid = m2id(m, i);
        r = frag2p(frag, rid);
        float2 s0 = __ldg(frag.pp + 0 + rid * 3);
        float2 s1 = __ldg(frag.pp + 1 + rid * 3);
        float2 s2 = __ldg(frag.pp + 2 + rid * 3);

        uint arg1 = mask ? lid : rid;
        uint arg2 = mask ? rid : lid;
        float myrandnr = l::rnd::d::mean0var1uu(seed, arg1, arg2);
        float3 r1 = make_float3(x, y, z), r2 = make_float3(r.x, r.y, r.z);
        float3 v1 = make_float3(vx, vy, vz), v2 = make_float3(s1.y, s2.x, s2.y);
        float3 strength = force(SOLVENT_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, myrandnr);

        xforce += strength.x;
        yforce += strength.y;
        zforce += strength.z;
    }
    atomicAdd(fx, xforce);
    atomicAdd(fy, yforce);
    atomicAdd(fz, zforce);
}

static __device__ void force1(const Frag frag, const Rnd rnd, /*const */ Pa p, /**/ Fo f) {
    int dx, dy, dz;
    Map m;
    m = p2map(frag, p.x, p.y, p.z);

    dx = frag.dx; dy = frag.dy; dz = frag.dz;
    p.x -= dx * XS;
    p.y -= dy * YS;
    p.z -= dz * ZS;
    force0(rnd, frag, m, p, f);
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

static __device__ void force2(const SFrag sfrag, const Frag frag, const Rnd rnd, uint i, /**/ float *ff) {
    Pa p;
    Fo f;
    p = sfrag2p(sfrag, i);
    f = sfrag2f(sfrag, ff, i);
    force1(frag, rnd, p, f);
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
    force2(sfrag, frag, rnd, i, /**/ ff);
}
}
