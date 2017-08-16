static __device__ float4 fetchH4(uint i) {
    return tex1Dfetch(texParticlesH4, i); /* (sic) type mismatch */
}

static __device__ float4 fetchF4(uint i) {
    return tex1Dfetch(texParticlesF4, i);
}
