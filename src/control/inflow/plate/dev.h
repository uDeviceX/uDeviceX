static __device__ void coords2pos(ParamsPlate p, float2 xi, /**/ float3 *r) {
    *r = p.o;
    axpy(xi.x, &p.a, /**/ r);
    axpy(xi.y, &p.b, /**/ r);
}

static __device__ float3 get_normal(ParamsPlate p, int2 nc, int i, int j) {
    float3 n;
    cross(&p.a, &p.b, /**/ &n);
    scal(1.f / (nc.x * nc.y), /**/ &n);
    return n;
}

static __device__ void coords2vel(VParamsPlate vp, ParamsPlate p, float2 xi, /**/ float3 *u) {
    float fact;
    fact = 1.f;
    if (vp.upoiseuille)
        fact *= 4 * xi.x * (1 - xi.x);

    if (vp.vpoiseuille)
        fact *= 4 * xi.y * (1 - xi.y);

    *u = vp.u;
    scal(fact, /**/ u);
}
