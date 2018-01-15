__device__ float3 tri(RbcParams_v par, float3 r1, float3 r2, float3 r3, Shape0 shape, float area, float volume) {
    float l0, A0, totArea;
    l0 = shape.a;
    A0 = shape.A;
    totArea = shape.totArea;
    return tri0(par, r1, r2, r3,   l0, A0, totArea,   area, volume);
}

__device__ float3 dih(RbcParams_v par, RbcParams_v par, float3 r0, float3 r1, float3 r2, float3 r3, float3 r4) {
    float3 f1, f2;
    f1 = dih0<1>(r0, r2, r1, r4);
    f2 = dih0<2>(r1, r0, r2, r3);
    add(&f1, /**/ &f2);
    return f2;
}
