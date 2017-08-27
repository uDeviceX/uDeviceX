namespace dev {

static __device__ int fetchS(int i) {
    return Ifetch(t::start, i);
}

static __device__ float2 fetchP(int i) {
    return F2fetch(t::pp, i);
}

}
