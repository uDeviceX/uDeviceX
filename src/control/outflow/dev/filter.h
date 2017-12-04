__global__ void filter(float3 origin, int n, const Particle *pp, Params params, /**/ int *ndead, int *kk) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int dead;
    Particle p;
    if (i >= n) return;

    p = pp[i];
    dead = predicate(origin, params, p.r);
    kk[i] = dead;
    if (dead)
        atomicAdd(ndead, 1);
    // if (dead)
    //     printf("%g %g %g   %g %g %g\n", p.r[0], p.r[1], p.r[2], p.v[0], p.v[1], p.v[2]);
}

