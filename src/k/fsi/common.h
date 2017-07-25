namespace k_fsi {
static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }

static __device__ void pair0(const Pa l, const Pa r, float rnd, /**/ float *fx, float *fy, float *fz) {
    /* pair force ; l, r: local and remote */
    float3 r1, r2, v1, v2, f;
    r1 = make_float3( l.x,  l.y,  l.z); r2 = make_float3( r.x,  r.y,  r.z);
    v1 = make_float3(l.vx, l.vy, l.vz); v2 = make_float3(r.vx, r.vy, r.vz);
    f = forces::dpd(SOLID_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, rnd); /* TODO: type */
    *fx = f.x; *fy = f.y; *fz = f.z;
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

static __device__ float random(uint lid, uint rid, float seed) {
    return rnd::mean0var1uu(seed, lid, rid);
}

static __device__ void tex2rv(int i,
                              float  *x, float  *y, float  *z,
                              float *vx, float *vy, float *vz) {
    float2 s0, s1, s2;
    i *= 3;
    s0 = tex1Dfetch(texSolventParticles, i++);
    s1 = tex1Dfetch(texSolventParticles, i++);
    s2 = tex1Dfetch(texSolventParticles, i++);

     *x = fst(s0);  *y = scn(s0);  *z = fst(s1);
    *vx = scn(s1); *vy = fst(s2); *vz = scn(s2);
}

static __device__ Pa tex2p(int i) {
    /* texture to particle */
    Pa p;
    tex2rv(i, /**/ &p.x, &p.y, &p.z,   &p.vx, &p.vy, &p.vz);
    return p;
}

static __device__ Fo ff2f(float *ff, int i) {
    Fo f;
    ff += 3*i;
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}

}
