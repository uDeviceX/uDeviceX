namespace dev {

static __device__ int fetchS(int i) {
    return Ifetch(t::start, i);
}

static __device__ float fetchP(int i) {
    return Ffetch(t::pp, i);
}

}
