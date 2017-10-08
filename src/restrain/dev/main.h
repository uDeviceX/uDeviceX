static __global__ void sum_vel(int color, int n, const Particle *pp, const int *cc) {
    int i, valid, nvalid;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    
    float3 v = make_float3(0, 0, 0);

    valid = (i < n) && (cc[i] == color);

    if (valid) {
        enum {X, Y, Z};
        const Particle p = pp[i]; 
        v.x = p.v[X];
        v.y = p.v[Y];
        v.z = p.v[Z];
    }

    v  = warpReduceSumf3(v);
    nvalid = warpReduceSum(valid);

    if ((threadIdx.x % warpSize == 0) && nvalid > 0) {
        atomicAdd(&g::v.x, v.x);
        atomicAdd(&g::v.y, v.y);
        atomicAdd(&g::v.z, v.z);
        atomicAdd(&g::n, nvalid);
    }
}

static __global__ void shift_vel(int color, float3 v, int n, const int *cc, /**/ Particle *pp) {
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
