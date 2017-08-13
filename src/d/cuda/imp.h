void ini() {
    /* panda specific for multi-gpu testing
       int device = m::rank % 2 ? 0 : 2; */
    int device = 0;
    CC(cudaSetDevice(device));
}
