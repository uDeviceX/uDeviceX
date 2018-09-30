struct Part { double3 r, v; };
typedef double3 Pos;

static __device__ Pos fetchPos(const Particle *pp, int i) {
    enum {X, Y, Z};
    Pos r;
    r.x = pp[i].r[X];
    r.y = pp[i].r[Y];
    r.z = pp[i].r[Z];
    return r;
}

static __device__ Part fetchPart(const Particle *pp, int i) {
    enum {X, Y, Z};
    Part p;
    const float *r, *v;
    r = pp[i].r; v = pp[i].v;
    p.r.x = r[X]; p.r.y = r[Y]; p.r.z = r[Z];
    p.v.x = v[X]; p.v.y = v[Y]; p.v.z = v[Z];
    return p;
}


__global__ void compute_area(int nt, int nc,
                             const Particle *pp, const int4 *tri,
                             /**/ float *area) {
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nt*nc) return;
}
