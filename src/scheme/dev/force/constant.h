__global__ void force(float mass, float driving_force0, int n, const Particle*, /**/ Force *ff) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *f = ff[pid].f;
    f[X] += mass*driving_force0;
}
