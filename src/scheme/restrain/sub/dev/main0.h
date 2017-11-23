static __device__ void sum0(int good, const Particle *pp, int i) {
    int ngood;
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
    ngood = warpReduceSum(good);

    if ((threadIdx.x % warpSize == 0) && ngood > 0) {
        atomicAdd(&g::v.x, v.x);
        atomicAdd(&g::v.y, v.y);
        atomicAdd(&g::v.z, v.z);
        atomicAdd(&g::n, ngood);
    }
}

static __device__ void shift0(int i, float3 v, /**/ Particle *pp) {
    enum {X, Y, Z};
    Particle p = pp[i];
    p.v[X] -= v.x; p.v[Y] -= v.y; p.v[Z] -= v.z;
    pp[i] = p;
}
