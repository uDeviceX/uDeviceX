static __device__ void coords2pos(float2 u, /**/ float3 *r) {
    
}

static __device__ float3 get_normal(int i, int j) {
    // TODO
    float3 n = make_float3(1,0,0);
    return n;
}

__global__ void cumulative_flux(int2 n, const float3 *flux, /**/ float *cumflux) {
    int i, xcid, ycid;
    float dn;
    float3 normal, f;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n.x * n.y) return;

    xcid = i % n.x;
    ycid = i / n.x;

    normal = get_normal(xcid, ycid);
    f      = flux[i];
    dn = dt * dot<float>(&normal, &f);

    cumflux[i] += dn;
}

static __device__ Particle create_particle(int xcid, int ycid, float3 f, curandState_t *rg) {
    Particle p = {0};
    return p;
}

__global__ void create_particles(int2 nc, const float3 *flux, /* io */ curandState_t *rnds, float *cumflux, /**/ int *n, Particle *pp) {
    int i, xcid, ycid, j, nnew, strt;
    float c;
    float3 f;
    curandState_t rndstate; 
    i = threadIdx.x + blockIdx.x * blockDim.x;
    xcid = i % nc.x;
    ycid = i / nc.x;

    if (i >= nc.x * nc.y) return;

    c = cumflux[i];
    nnew = int(c);

    if (nnew == 0) return;

    c -= nnew;
    cumflux[i] = c;

    f        = flux[i];
    rndstate = rnds[i];
    
    strt = atomicAdd(n, nnew);
    
    for (j = strt; j < strt + nnew; ++j) {
        pp[j] = create_particle(xcid, ycid, f, /*io*/ &rndstate);
    }
    rnds[i] = rndstate;
}
