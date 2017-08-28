namespace odstr { namespace sub { namespace dev {
/* utils */
__device__ void xchg(int dw, /**/ float3 *s0, float3 *s1) { /* collective */
    int src0, src1;
    if (dw % 2  == 0) {
        src0 =      dw / 2; src1 = 16 + dw / 2;
    } else {
        src0 = 16 + dw / 2; src1 = dw / 2;
    }
    xchg_aos4f(src0, src1, dw % 2, /**/ s0, s1); /* collective */
}

__device__ void zip(float r[3], float v[3], int ws, int dw, int dwe, /**/
                    float4  *zip0, ushort4 *zip1) { /* collective */
    enum {X, Y, Z};
    float3 s0, s1;    
    s0 = make_float3(r[X], r[Y], r[Z]);
    s1 = make_float3(v[X], v[Y], v[Z]);
    xchg(dw, &s0, &s1); /* collective */
    if (dw < 2 * dwe)
        zip0[2 * ws + dw] = make_float4(s0.x, s0.y, s0.z, 0);
    if (dw + 32 < 2 * dwe)
        zip0[2 * ws + dw + 32] = make_float4(s1.x, s1.y, s1.z, 0);
    if (dw < dwe)
        zip1[ws + dw] = make_ushort4(__float2half_rn(r[X]),
                                     __float2half_rn(r[Y]),
                                     __float2half_rn(r[Z]),
                                     0);
}

static __device__ int x2c(float x, int L) {
    int i = (int) floor((double)x + L / 2);
    check_cel(x, i, L);
    return i;
}

__device__ void r2c(float r[3], /**/ int* ix, int* iy, int* iz, int* i) {
    /* position to cell coordinates */
    enum {X, Y, Z};
    int x, y, z;
    x = x2c(r[X], XS);
    y = x2c(r[Y], YS);
    z = x2c(r[Z], ZS);
    *i  = x + XS * (y + YS * z);

    *ix = x; *iy = y; *iz = z;
}

}}} /* namespace */
