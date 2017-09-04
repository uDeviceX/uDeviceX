static __device__ float4 fetchH4(uint i) {
    return F4fetch(texParticlesH4, i); /* (sic) type mismatch */
}

static __device__ float4 fetchF4(uint i) {
    return F4fetch(texParticlesF4, i);
}

static __device__ int fetchC(uint i) {
    return Ifetch(texColor, i);
}
