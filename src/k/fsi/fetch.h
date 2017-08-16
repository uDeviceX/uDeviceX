namespace k_fsi {

static __device__ int fetchS(int i) {
    return tex1Dfetch(t::start, i);
}

static __device__ float2 fetchP(int i) {
    return tex1Dfetch(t::pp, i);
}

}
