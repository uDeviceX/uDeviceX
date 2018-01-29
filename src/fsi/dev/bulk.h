static __device__ void p2rv(const float *p, int i, /**/
                            float  *x, float  *y, float  *z,
                            float *vx, float *vy, float *vz) {
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    i *= 6;
     *x = p[i++];  *y = p[i++];  *z = p[i++];
    *vx = p[i++]; *vy = p[i++]; *vz = p[i++];
}

static __device__ void pp2p(const float *pp, int i, /**/ Pa *p) { /* TODO gets force::Pa directly */
    p2rv(pp, i, /**/ &p->x, &p->y, &p->z,   &p->vx, &p->vy, &p->vz);
    p->kind = SOLID_KIND;
}

static __device__ int p2map(int3 L, const int *start, int zplane, int n, const Pa p, /**/ Map *m) {
    /* particle to map */
    return r2map(L, start, zplane, n, p.x, p.y, p.z, m);
}

const static float EPS = 1e-6;
static __device__ float dist(Pa a, Pa b) {
    float dx, dy, dz;
    dx = a.x - b.x;
    dy = a.y - b.y;
    dz = a.z - b.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

static __device__ void bulk0(const Pa a, Cloud cloud, int lid, const Map m, float seed, /**/
                             float *fx, float *fy, float *fz, float *ff) {
    /* "[a]ocal" and "[b]emote" particles */
    Pa b;
    Fo f;
    int i, rid;
    *fx = *fy = *fz = 0; /* local force */
    for (i = 0; !endp(m, i); ++i) {
        rid = m2id(m, i);
        cloud_get(cloud, rid, /**/ &b);
        f = ff2f(ff, rid);
        pair(a, b, random(lid, rid, seed), /**/ fx, fy, fz,   f);
    }
}

static __device__ void bulk1(const Pa a, Cloud cloud,
                             const Fo f, int i, const Map m, float seed, /**/ float *ff) {
    float fx, fy, fz; /* local force */
    bulk0(a, cloud, i, m, seed, /**/ &fx, &fy, &fz, ff);
    atomicAdd(f.x, fx);
    atomicAdd(f.y, fy);
    atomicAdd(f.z, fz);
}

static __device__ void bulk2(int3 L, const int *start, float *ppA, Cloud cloud, int i, int zplane, int n, float seed,
                             /**/ float *ff, float *ff0) {
    Pa p;
    Fo f; /* "local" particle */
    Map m;
    pp2p(ppA, i, /**/ &p);
    f = ff2f(ff, i);
    if (!p2map(L, start, zplane, n, p, /**/ &m)) return;
    bulk1(p, cloud, f, i, m, seed, /**/ ff0);
}

__global__ void bulk(int3 L, const int *start, float *ppA, Cloud cloud, int n0, int n1, float seed, float *ff, float *ff0) {
    int gid, i, zplane;
    gid    = threadIdx.x + blockDim.x * blockIdx.x;
    i      = gid / 3;
    zplane = gid % 3;
    if (i >= n0) return;
    bulk2(L, start, ppA, cloud, i, zplane, n1, seed, ff, ff0);
}
