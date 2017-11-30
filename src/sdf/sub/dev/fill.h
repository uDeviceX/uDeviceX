__global__ static void fill(const tex3Dca texsdf, const Particle *const pp, const int n,
                            int *const key) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = pp[pid];
    float sdf0 = sdf(texsdf, p.r[X], p.r[Y], p.r[Z]);
    key[pid] = (int)(sdf0 >= 0) + (int)(sdf0 > 2);
}
