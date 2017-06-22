namespace k_sim {

__global__ void update(float mass, Particle* pp, Force* ff, int n) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    float *v = pp[pid].v, *r = pp[pid].r, *f = ff[pid].f;
  
#ifdef FORWARD_EULER
    r[0] += v[0]*dt;
    r[1] += v[1]*dt;
    r[2] += v[2]*dt;

    v[0] += f[0]/mass*dt;
    v[1] += f[1]/mass*dt;
    v[2] += f[2]/mass*dt;
#else // velocity verlet
    v[0] += f[0]/mass*dt;
    v[1] += f[1]/mass*dt;
    v[2] += f[2]/mass*dt;

    r[0] += v[0]*dt;
    r[1] += v[1]*dt;
    r[2] += v[2]*dt;
#endif
}

__global__ void body_force(float mass, Particle* pp, Force* ff, int n, float driving_force0) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    float *r = pp[pid].r, *f = ff[pid].f;

    float dy = r[1] - glb::r0[1]; /* coordinate relative to domain
                                     center */
    if (doublepoiseuille && dy <= 0) driving_force0 *= -1;
    f[0] += mass*driving_force0;
}

__global__ void clear_velocity(Particle *pp, int n)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}

__global__ void ic_shear_velocity(Particle *pp, int n)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float z = pp[pid].r[2] - glb::r0[2];
    float vx = gamma_dot*z, vy = 0, vz = 0;
    pp[pid].v[0] = vx; pp[pid].v[1] = vy; pp[pid].v[2] = vz;
}

} /* end of k_sim */
