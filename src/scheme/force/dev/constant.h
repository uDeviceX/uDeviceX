__global__ void main(float mass, Param par, int n, const Particle*, /**/ Force *ff) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *f = ff[pid].f;
    f[X] += mass * par.a;
    f[Y] += mass * par.b;
    f[Z] += mass * par.c;
}
