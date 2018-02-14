namespace dev {
__global__ void zip(int  n, const float *pp, /**/ float4 *zpp) {
    enum {X, Y, Z};
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    int i;
    const float *r, *v;
    float x, y, z;
    float vx, vy, vz;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    r = &pp[6*i];
    v = &pp[6*i + 3];

    x =  r[X];  y = r[Y];  z = r[Z];
    vx = v[X]; vy = v[Y]; vz = v[Z];

    zpp[2*i]     = make_float4(x,   y,  z, 0);
    zpp[2*i + 1] = make_float4(vx, vy, vz, 0);
}

} /* namespace */
