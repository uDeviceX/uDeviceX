static __device__ int fetchS(int i) {
    return Ifetch(c::start, i);
}

static __device__ int fetchID(int i) {
    return Ifetch(c::id, i);
}
