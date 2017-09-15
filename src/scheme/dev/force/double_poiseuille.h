__global__ void force(float mass, Particle* pp, Force* ff, int n, float driving_force0) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    float *r = pp[pid].r, *f = ff[pid].f;

    float d = r[Y] - glb::r0[Y]; /* coordinate relative to domain
                                     center */
    if (d <= 0) driving_force0 *= -1;
    f[X] += mass*driving_force0;
}

