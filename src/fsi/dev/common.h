namespace dev {
static __device__ float fst(float2 p) { return p.x; }
static __device__ float scn(float2 p) { return p.y; }


static __device__ float random(uint lid, uint rid, float seed) {
    return rnd::mean0var1uu(seed, lid, rid);
}

static __device__ void tex2rv(int i,
                              float  *x, float  *y, float  *z,
                              float *vx, float *vy, float *vz) {
    i *= 6;
     *x = fetchP(i++);  *y = fetchP(i++);  *z = fetchP(i++);
    *vx = fetchP(i++); *vy = fetchP(i++); *vz = fetchP(i++);
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
