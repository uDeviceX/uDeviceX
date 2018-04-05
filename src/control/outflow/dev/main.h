static __device__ int predicate(ParamsPlate p, const float r[3]) {
    enum {X, Y, Z};
    float s;
    
    s = p.a * r[X] + p.b * r[Y] + p.c * r[Z] + p.d;
    
    return s > 0;
}

static __device__ int predicate(ParamsCircle p, const float r[3]) {
    enum {X, Y, Z};
    float x, y, rsq;
    x = r[X] - p.c.x;
    y = r[Y] - p.c.y;

    rsq = x*x + y*y;
    return rsq > p.Rsq;
}

template <typename Par>
__global__ void filter(int n, const Particle *pp, Par params, /**/ int *ndead, int *kk) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int dead;
    Particle p;
    if (i >= n) return;

    p = pp[i];
    dead = predicate(params, p.r);
    kk[i] = dead;
    if (dead)
        atomicAdd(ndead, 1);
}

