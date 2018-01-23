namespace dev {
__device__ ushort f2s(float x) { return __float2half_rn(x); }
__global__ void zip(int  n, const float *pp, /**/ float4 *zip0, ushort4 *zip1) {
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

    zip0[2*i]     = make_float4(x,   y,  z, 0);
    zip0[2*i + 1] = make_float4(vx, vy, vz, 0);
    zip1[i] = make_ushort4(f2s(x), f2s(y), f2s(z), 0);
}

} /* namespace */
