static __device__ void cloud_get_common(Cloud c, int i, /**/ PairPa *p) {
    const float2 *pp = (const float2 *) c.pp;
    float2 s0, s1, s2;
    s0 = __ldg(pp + 3*i + 0);
    s1 = __ldg(pp + 3*i + 1);
    s2 = __ldg(pp + 3*i + 2);

    p->x = s0.x;
    p->y = s0.y;
    p->z = s1.x;

    p->vx = s1.y;
    p->vy = s2.x;
    p->vz = s2.y;
}

static __device__ void cloud_get_p(Cloud c, int i, /**/ PairPa *p) {
    cloud_get_common(c, i, /**/ p);
}

static __device__ void cloud_get(Cloud c, int i, /**/ PairPa *p) {
    cloud_get_common(c, i, /**/ p);
    if (multi_solvent) p->color = c.cc[i];
}
