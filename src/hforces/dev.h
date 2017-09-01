namespace hforces {
namespace dev {

struct Pa { /* local particle */
    float x, y, z;
    float vx, vy, vz;
    int id;
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

static __device__ Pa frag2p(const Frag frag, int i) {
    Pa p;
    p2rv2(frag.pp, i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    p.id = i;
    return p;
}

static __device__ void pair(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    forces::Pa a, b;
    forces::r3v3k2p(l.x, l.y, l.z, l.vx, l.vy, l.vz, SOLVENT_TYPE, /**/ &a);
    forces::r3v3k2p(r.x, r.y, r.z, r.vx, r.vy, r.vz, SOLVENT_TYPE, /**/ &b);
    forces::gen(a, b, rnd, /**/ fx, fy, fz);
}

static __device__ float random(int lid, uint rid, float seed, int mask) {
    uint a1, a2;
    a1 = mask ? lid : rid;
    a2 = mask ? rid : lid;
    return rnd::mean0var1uu(seed, a1, a2);
}

static __device__ void force0(const Rnd rnd, const Frag frag, const Map m, const Pa l, /**/
                              float *fx, float *fy, float *fz) {
    /* l, r: local and remote particles */
    Pa r;
    int i;
    int lid, rid; /* ids */
    float x, y, z; /* pair force */
    lid = l.id;

    *fx = *fy = *fz = 0;
    for (i = 0; !endp(m, i); i ++ ) {
        rid = m2id(m, i);
        r = frag2p(frag, rid);
        pair(l, r, random(lid, rid, rnd.seed, rnd.mask), &x, &y, &z);
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
    m = r2map(frag, p.x, p.y, p.z);

    dx = frag.dx; dy = frag.dy; dz = frag.dz; /* TODO: where it should be? */
    p.x -= dx * XS;
    p.y -= dy * YS;
    p.z -= dz * ZS;
    force1(rnd, frag, m, p, f);
}

static __device__ Fo i2f(const int *ii, float *ff, int i) {
    /* local id and index to force */
    Fo f;
    ff += 3*ii[i];
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

static __device__ void p2rv(const float *p, int i,
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    p += 6*i;
     *x = *(p++);  *y = *(p++);  *z = *(p++);
    *vx = *(p++); *vy = *(p++); *vz = *(p++);
}

static __device__ Pa sfrag2p(const SFrag sfrag, int i) {
    Pa p;
    p2rv(sfrag.pp,     i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    p.id = i;
    return p;
}

static __device__ Fo sfrag2f(const SFrag sfrag, float *ff, int i) {
    return i2f(sfrag.ii, ff, i);
}

static __device__ void force3(const SFrag sfrag, const Frag frag, const Rnd rnd, int i, /**/ float *ff) {
    Pa p;
    Fo f;
    p = sfrag2p(sfrag, i);
    f = sfrag2f(sfrag, ff, i);
    force2(frag, rnd, p, f);
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

} // dev
} // hforces
