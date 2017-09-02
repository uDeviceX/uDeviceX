namespace dev {

static __device__ int fetchS(int i) {
    return Ifetch(t::start, i);
}

}
