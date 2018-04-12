#define _S_ static __device__
#define _I_ static __device__

template <typename Datum, typename Pa>
_S_ void part(int i, const Datum *d, Pa *p) {
    float2 p0, p1, p2;
    const float2 *pp = (const float2*) d->pp;
    p0 = pp[3*i+0]; p1 = pp[3*i+1]; p2 = pp[3*i+2];

    p->r = make_float3(p0.x, p0.y, p1.x);
    p->v = make_float3(p1.y, p2.x, p2.y);
}

_S_ void stress(int i, const DatumS_v *d, PartS *p) {
    float2 s0, s1, s2;
    const float2 *ss = (const float2*) d->ss;
    s0 = ss[3*i+0]; s1 = ss[3*i+1]; s2 = ss[3*i+2];

    p->s1 = make_float3(s0.x, s0.y, s1.x);
    p->s2 = make_float3(s1.y, s2.x, s2.y);
}

_I_ Part fetch_part(int i, const Datum_v *d) {
    Part p;
    part(i, d, &p);
    return p;
}

_I_ PartS fetch_part(int i, const DatumS_v *d) {
    PartS p;
    part(i, d, &p);
    stress(i, d, &p);
    return p;
}

#undef _S_
#undef _I_
