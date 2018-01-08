static __device__ void coords2pos(Params p, float2 xi, /**/ float3 *r) {
    float th, h, cth, sth;
    th = xi.x * 2 * M_PI;
    h = xi.y * p.H;
    cth = cos(th);
    sth = sin(th);

    *r = p.o;
    r->x += p.R * cth;
    r->y += p.R * sth;
    r->z += h;
}

static __device__ float3 get_normal(Params p, int2 nc, int i, int j) {
    float3 n;
    float th, cth, sth, area;
    th = (i + 0.5f) / (nc.x) * 2 * M_PI;
    cth = cos(th);
    sth = sin(th);

    n = make_float3(cth, sth, 0);
    area = p.H * p.R * 2 * M_PI / (nc.x * nc.y);
    scal(area, /**/ &n);
    return n;
}

static __device__ void coords2vel(VParams vp, Params p, float2 xi, /**/ float3 *u) {
    float th, cth, sth, fact;
    th = xi.x * 2 * M_PI;
    cth = cos(th);
    sth = sin(th);

    fact = 1.f;
    if (vp.poiseuille) {
        float y = xi.y;
        fact *= 4 * y * (1 - y);
    }

    *u = make_float3(cth, sth, 0);
    scal(fact * vp.u, /**/ u);
}
