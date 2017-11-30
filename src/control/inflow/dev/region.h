static void coords2pos(float2 u, /**/ float3 *r) {
    
}

static float3 get_normal(int i, int j) {
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
    dn = dt * dot<float>(normal, f);

    cumflux[i] += dn;
}


__global__ void create_particles(int2 n, const float3 *flux, /* io */ float *cumflux, /**/ int *n, Particle *pp) {
    int i, xcid, ycid, j, nnew, strt;
    float c;
    Particle p;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    xcid = i % n.x;
    ycid = i / n.x;

    if (i >= n.x * n.y) return;

    c = cumflux[i];
    nnew = int(c);

    if (nnew == 0) return;

    c -= nnew;
    cumflux[i] = c;
    
    strt = atomicAdd(n, nnew);

    for (j = strt; j < strt + nnew; ++j) {
        pp[j] = create_particle(xcid, ycid, f);
    }
}
