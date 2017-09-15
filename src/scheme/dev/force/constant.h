__global__ void force(float mass, Particle*, Force *ff, int n, float driving_force0) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *f = ff[pid].f;
    f[X] += mass*driving_force0;
}
