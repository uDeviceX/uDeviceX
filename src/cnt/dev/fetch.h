static __device__ int fetchS(int i) {
    return Ifetch(t::start, i);
}

static __device__ int fetchID(int i) {
    return Ifetch(t::id, i);
}
