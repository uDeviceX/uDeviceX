static __device__ float random(uint lid, uint rid, float seed) {
    return rnd::mean0var1uu(seed, lid, rid);
}
static __device__ Fo ff2f(float *ff, int i) {
    Fo f;
    ff += 3*i;
    f.x = ff++; f.y = ff++; f.z = ff++;
    return f;
}
