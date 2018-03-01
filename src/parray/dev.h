#define _S_ static __device__
#define _I_ static __device__

template<typename PA>
_S_ void parray_get_common(PA a, int i, /**/ PairPa *p) {
    const float2 *pp = (const float2 *) a.pp;
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

// tag::int[]
_I_ void parray_get(PaArray_v a, int i, /**/ PairPa *p) // <1>
// end::int[]
{
    parray_get_common(a, i, /**/ p);
}

// tag::int[]
_I_ void parray_get(PaCArray_v a, int i, /**/ PairPa *p) // <2>
// end::int[]
{
    parray_get_common(a, i, /**/ p);
    p->color = a.cc[i];
}

#undef _S_
#undef _I_
