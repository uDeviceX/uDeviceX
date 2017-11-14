__device__ float3 tri(float3 r1, float3 r2, float3 r3, float area, float volume) {
    return tri0(r1, r2, r3, area, volume);
}

template <int update>
__device__ float3 dih(float3 r1, float3 r2, float3 r3, float3 r4) {
    return dih0<update>(r1, r2, r3, r4);
}
