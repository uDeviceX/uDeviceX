__global__ void update(float mass, Particle* pp, Force* ff, int n) {
    float *r, *v, *f;
    int pid;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    v = pp[pid].v;
    f = ff[pid].f;
    update0(mass, f, /**/ r, v);
}

__global__ void clear_vel(Particle *pp, int n)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}
