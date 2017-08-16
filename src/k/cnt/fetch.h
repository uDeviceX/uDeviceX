namespace k_cnt {

static __device__ int fetchS(int i) {
    return tex1Dfetch(t::start, i);
}

static __device__ int fetchID(int i) {
    return tex1Dfetch(t::id, i);
}


}
