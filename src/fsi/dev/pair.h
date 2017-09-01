namespace dev {
static __device__ void pair0(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* pair force ; l, r: local and remote */
    forces::Pa a, b;
    forces::r3v3k2p(l.x, l.y, l.z, l.vx, l.vy, l.vz, SOLID_TYPE, /**/ &a);
    forces::r3v3k2p(r.x, r.y, r.z, r.vx, r.vy, r.vz, SOLID_TYPE, /**/ &b);
    forces::gen(a, b, rnd, /**/ fx, fy, fz);
}

static __device__ void pair(const Pa l, const Pa r, float rnd, /**/
                            float *fx, float *fy, float *fz,
                            Fo f) {
    /* f[xyz]: local force; Fo f: remote force */
    float x, y, z; /* pair force */
    pair0(l, r, rnd, /**/ &x, &y, &z);
    *fx += x; *fy += y; *fz += z;
    atomicAdd(f.x, -x); atomicAdd(f.y, -y); atomicAdd(f.z, -z);
}
}
