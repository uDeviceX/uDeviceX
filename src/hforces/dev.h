namespace hforces { namespace dev {

struct PB {
    float x, y, z;
    float vx, vy, vz;
};

struct Fo { float *x, *y, *z; }; /* force */

static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }
static __device__ void p2rv2(const float2 *p, int i,
                             float  *x, float  *y, float  *z,
                             float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    p += 3*i;
    s0 = __ldg(p++); s1 = __ldg(p++); s2 = __ldg(p++);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ PB frag2p(const Frag frag, int i) {
    PB p;
    p2rv2(frag.pp, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

static __device__ void pair(const forces::Pa a, const PB b, float rnd, /**/ float *fx, float *fy, float *fz) {
    forces::Pa b0;
    forces::r3v3k2p(b.x, b.y, b.z, b.vx, b.vy, b.vz, SOLVENT_TYPE, /**/ &b0);
    forces::gen(a, b0, rnd, /**/ fx, fy, fz);
}

static __device__ float random(int aid, int bid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? aid : bid;
    a2 = mask ? bid : aid;
    return rnd::mean0var1uu(seed, a1, a2);
}

static __device__ void force0(const Rnd rnd, const Frag frag, const Map m, const forces::Pa a, int aid, /**/
                              float *fx, float *fy, float *fz) {
    PB b;
    int i;
    int bid; /* ids */
    float x, y, z; /* pair force */

    *fx = *fy = *fz = 0;
    for (i = 0; !endp(m, i); i ++ ) {
        bid = m2id(m, i);
        b = frag2p(frag, bid);
        pair(a, b, random(aid, bid, rnd.seed, rnd.mask), &x, &y, &z);
        *fx += x; *fy += y; *fz += z;
    }
}

static __device__ void force1(const Rnd rnd, const Frag frag, const Map m, const forces::Pa p, int id, Fo f) {
    float x, y, z; /* force */
    force0(rnd, frag, m, p, id, /**/ &x, &y, &z);
    atomicAdd(f.x, x);
    atomicAdd(f.y, y);
    atomicAdd(f.z, z);
}

static __device__ void force2(const Frag frag, const Rnd rnd, forces::Pa p, int id, /**/ Fo f) {
    int dx, dy, dz;
    Map m;
    m = r2map(frag, p.x, p.y, p.z);

    dx = frag.dx; dy = frag.dy; dz = frag.dz; /* TODO: where it should be? */
    p.x -= dx * XS;
    p.y -= dy * YS;
    p.z -= dz * ZS;
    force1(rnd, frag, m, p, id, f);
}

static __device__ Fo i2f(const int *ii, float *ff, int i) {
    /* local id and index to force */
    Fo f;
    ff += 3*ii[i];
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

static __device__ Fo sfrag2f(const SFrag sfrag, float *ff, int i) {
    return i2f(sfrag.ii, ff, i);
}

static __device__ void force3(const SFrag sfrag, const Frag frag, const Rnd rnd, int i, /**/ float *ff) {
    forces::Pa p;
    Fo f;
    cloudA_get(sfrag.c, i, &p);
    f = sfrag2f(sfrag, ff, i);
    force2(frag, rnd, p, i, f);
}

__global__ void force(const int27 start, const SFrag26 ssfrag, const Frag26 ffrag, const Rnd26 rrnd, /**/ float *ff) {
    Frag frag;
    Rnd  rnd;
    SFrag sfrag;
    int gid;
    int fid; /* fragment id */
    int i; /* particle id */

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= start.d[26]) return;
    fid = k_common::fid(start.d, gid);
    i = gid - start.d[fid];
    sfrag = ssfrag.d[fid];
    if (i >= sfrag.n) return;

    frag = ffrag.d[fid];
    rnd = rrnd.d[fid];
    force3(sfrag, frag, rnd, i, /**/ ff);
}

}} /* namespace */
