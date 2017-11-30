__global__ void filter(float3 origin, int n, const Particle *pp, Params params, /**/ Outflow o) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int dead;
    Particle p;
    if (i >= n) return;

    p = pp[i];
    dead = predicate(origin, params, p.r);
    o.kk[i] = dead;
    if (dead)
        atomicAdd(o.ndead_dev, 1);
}

