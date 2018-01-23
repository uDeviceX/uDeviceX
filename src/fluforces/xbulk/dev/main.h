__device__ void fetch(BCloud c, int i, forces::Pa *p) {
    float4 r, v;
    r = c.pp[2*i + 0];
    v = c.pp[2*i + 1];

    forces::r3v3k2p(r.x, r.y, r.z,
                    v.x, v.y, v.z,
                    SOLVENT_KIND, /**/ p);
    
    if (multi_solvent)
        p->color = c.cc[i];
}

__global__ void apply(int n, BCloud cloud, const int *start, const int *count, RNDunif *rnd, /**/ Force *ff) {
    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    
}
