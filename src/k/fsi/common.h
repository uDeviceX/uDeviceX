namespace k_fsi {
static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }

static __device__ float random(uint lid, uint rid, float seed) {
    return l::rnd::d::mean0var1uu(seed, lid, rid);
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
