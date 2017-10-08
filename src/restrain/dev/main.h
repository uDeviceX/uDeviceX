static __device__ void sum0(int good, const Particle *pp, int i) {
    int nvalid;
    float3 v;
    
    if (good) {
        enum {X, Y, Z};
        const Particle p = pp[i]; 
        v.x = p.v[X];
        v.y = p.v[Y];
        v.z = p.v[Z];
    } else {
        v.x = v.y = v.z = 0;
    }

    v  = warpReduceSumf3(v);
    nvalid = warpReduceSum(good);

    if ((threadIdx.x % warpSize == 0) && nvalid > 0) {
        atomicAdd(&g::v.x, v.x);
        atomicAdd(&g::v.y, v.y);
        atomicAdd(&g::v.z, v.z);
        atomicAdd(&g::n, nvalid);
    }
}

static __global__ void sum(int color, int n, const Particle *pp, const int *cc) {
    int i, good;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    good = (i < n) && (cc[i] == color);
    sum0(good, pp, i);
}

static __global__ void shift(int color, float3 v, int n, const int *cc, /**/ Particle *pp) {
    enum {X, Y, Z};
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= n || cc[i] != color) return;

    Particle p = pp[i];
    
    p.v[X] -= v.x;
    p.v[Y] -= v.y;
    p.v[Z] -= v.z;

    pp[i] = p;
}
