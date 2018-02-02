__global__ void update(float dt0, MoveParams_v parv, float mass, Particle* pp, const Force *ff, int n) {
    float *r, *v;
    const float *f;
    int pid;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    v = pp[pid].v;
    f = ff[pid].f;
    update0(dt0, parv, mass, f, /**/ r, v);
}

__global__ void clear_vel(int n, /**/ Particle *pp)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}
