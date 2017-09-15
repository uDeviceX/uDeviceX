__global__ void force(float mass, Particle *pp, Force *ff, int, float) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

}
