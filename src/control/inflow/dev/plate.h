struct Params {
    float3 o, a, b;
};

static __device__ void coords2pos(Params p, float2 u, /**/ float3 *r) {
    *r = p.o;
    axpy(u.x, &p.a, /**/ r);
    axpy(u.y, &p.b, /**/ r);
}

static __device__ float3 get_normal(Params p, int2 nc, int i, int j) {
    float3 n;
    cross(&p.a, &p.b, /**/ &n);
    scal(1.f / (nc.x * nc.y), /**/ &n);
    return n;
}
