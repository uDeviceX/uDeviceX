template <typename Par, typename VPar>
__global__ void ini_vel(VPar vparams, Par params, int2 nc, /**/ float3 *uu) {
    int i, xcid, ycid;
    float3 u;
    float2 xi;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nc.x * nc.y) return;

    xcid = i % nc.x;
    ycid = i / nc.x;

    xi.x = (xcid + 0.5f) / nc.x;
    xi.y = (ycid + 0.5f) / nc.y;

    coords2vel(vparams, params, xi, /**/ &u);
    
    uu[i] = u;
}

template <typename Par>
__global__ void cumulative_flux(Par params, int2 nc, const float3 *uu, /**/ float *cumflux) {
    int i, xcid, ycid;
    float dn;
    float3 normal, u;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nc.x * nc.y) return;

    xcid = i % nc.x;
    ycid = i / nc.x;

    // normal is scaled by area of element
    normal = get_normal(params, nc, xcid, ycid);
    u      = uu[i];
    dn = dt * numberdensity * dot<float>(&normal, &u);

    cumflux[i] += dn;
}

template <typename Par>
static __device__ Particle create_particle(Par params, int2 nc, int xcid, int ycid, float3 u, curandState_t *rg) {
    float2 xi;
    float3 r;
    float sigma;
    xi.x = (xcid + curand_uniform(rg)) / nc.x;
    xi.y = (ycid + curand_uniform(rg)) / nc.y;

    coords2pos(params, xi, /**/ &r);

    sigma = sqrtf(kBT);
    u.x += curand_normal(rg) * sigma;
    u.y += curand_normal(rg) * sigma;
    u.z += curand_normal(rg) * sigma;

    // printf("create one particle: %g %g %g   %g %g %g\n", r.x, r.y, r.z, u.x, u.y, u.z);
    
    return Particle({r.x, r.y, r.z, u.x, u.y, u.z});
}

template <typename Par>
__global__ void create_particles(Par params, int2 nc, const float3 *flux, /* io */ curandState_t *rnds, float *cumflux, /**/ int *n, SolventWrap wrap) {
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
        wrap.pp[j] = create_particle(params, nc, xcid, ycid, f, /*io*/ &rndstate);
        if (wrap.multisolvent)
            wrap.cc[j] = 0; // TODO
    }

    rnds[i] = rndstate;
}
