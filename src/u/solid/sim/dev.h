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

__global__ void body_force(float mass, Particle* pp, Force* ff, int n, float driving_force0) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    float *r = pp[pid].r, *f = ff[pid].f;

    float d = r[Y] - glb::r0[Y]; /* coordinate relative to domain
                                     center */
    if (doublepoiseuille && d <= 0) driving_force0 *= -1;
    f[X] += mass*driving_force0;
}

__global__ void clear_vel(Particle *pp, int n)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}
