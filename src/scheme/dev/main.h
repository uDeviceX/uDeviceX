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

__global__ void clear_vel(Particle *pp, int n)  {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    for(int c = 0; c < 3; ++c) pp[pid].v[c] = 0;
}

namespace restrain_drop {

__device__ float3 sumv;
__device__ int indrop;

static __device__ float3 warpReduceSumf3(float3 v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
        v.x += __shfl_down(v.x, offset);
        v.x += __shfl_down(v.y, offset);
        v.z += __shfl_down(v.z, offset);
    }
    return v;
}

static __device__ int warpReduceSum(int v) {
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

__global__ void reduce_vel(int color, int n, const Particle *pp, const int *cc) {
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

    if (threadIdx.x == 0) {
        atomicAdd(&sumv.x, v.x);
        atomicAdd(&sumv.y, v.y);
        atomicAdd(&sumv.z, v.z);
        atomicAdd(&indrop, nvalid);
    }
}

} // restrain_drop
