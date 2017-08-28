namespace dev {
static __device__ float2 get(const float2 *p) { return __ldg(p); }
static __device__ void p2rv(const float2 *p, int i, /**/
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    p += 3*i;
    s0 = get(p++); s1 = get(p++); s2 = get(p++);
     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);

    bool verbose = true;
    assert(dbg::dev::valid_unpacked_pos_pu(*x, *y, *z, verbose));
    assert(dbg::dev::valid_vel3(*vx, *vy, *vz, verbose));
}

static __device__ void pp2p(float2 *pp, int i, /**/ Pa *p) {
    p2rv(pp, i, /**/ &p->x, &p->y, &p->z,   &p->vx, &p->vy, &p->vz);
}

static __device__ int p2map(int zplane, int n, const Pa p, /**/ Map *m) {
    /* particle to map */
    return r2map(zplane, n, p.x, p.y, p.z, m);
}

static __device__ void bulk0(const Pa l, int lid, const Map m, float seed, /**/
                             float *fx, float *fy, float *fz, float *ff) {
    /* "[l]ocal" and "[r]emote" particles */
    Pa r;
    Fo f;
    int i, rid;

    *fx = *fy = *fz = 0; /* local force */
    for (i = 0; !endp(m, i); ++i) {
        rid = m2id(m, i);
        r = tex2p(rid);
        f = ff2f(ff, rid);
        pair(l, r, random(lid, rid, seed), /**/ fx, fy, fz,   f);
    }
}

static __device__ void bulk1(const Pa l, const Fo f, int i, const Map m, float seed, /**/ float *ff) {
    float fx, fy, fz; /* "local" force */
    bulk0(l, i, m, seed, /**/ &fx, &fy, &fz, ff);
    atomicAdd(f.x, fx);
    atomicAdd(f.y, fy);
    atomicAdd(f.z, fz);
}

static __device__ void bulk2(float2 *pp, int i, int zplane, int n, float seed, /**/ float *ff, float *ff0) {
    Pa p;
    Fo f; /* "local" particle */
    Map m;
    pp2p(pp, i, /**/ &p);
    f = ff2f(ff, i);
    if (!p2map(zplane, n, p, /**/ &m)) return;
    bulk1(p, f, i, m, seed, /**/ ff0);
}

__global__ void bulk(float2 *pp, int n0, int n1, float seed, float *ff, float *ff0) {
    int gid, i, zplane;
    gid    = threadIdx.x + blockDim.x * blockIdx.x;
    i      = gid / 3;
    zplane = gid % 3;
    if (i >= n0) return;
    bulk2(pp, i, zplane, n1, seed, ff, ff0);
}
}
