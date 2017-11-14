__device__ float3 tri(float3 r1, float3 r2, float3 r3, Shape0 shape, float area, float volume) {
    float l0, A0;
    A0 = RBCtotArea / (2.0 * RBCnv - 4.);
    l0 = sqrt(A0 * 4.0 / sqrt(3.0));
    return tri0(r1, r2, r3,   l0, A0,   area, volume);
}

__device__ float3 dih(float3 r0, float3 r1, float3 r2, float3 r3, float3 r4) {
    float3 f1, f2;
    f1 = dih0<1>(r0, r2, r1, r4);
    f2 = dih0<2>(r1, r0, r2, r3);
    add(&f1, /**/ &f2);
    return f2;
}
