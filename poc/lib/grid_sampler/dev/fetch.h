#define _S_ static __device__
#define _I_ static __device__

_I_ void fetch_part(int i, const SampleDatum *d, Part *p) {
    float2 p0, p1, p2;
    const float2 *pp = (const float2*) d->pp;
    p0 = pp[3*i+0]; p1 = pp[3*i+1]; p2 = pp[3*i+2];

    p->r = make_float3(p0.x, p0.y, p1.x);
    p->v = make_float3(p1.y, p2.x, p2.y);
}

_I_ void fetch_color(int i, const SampleDatum *d, int *c) {
    *c = d->cc[i];
}

_I_ void fetch_stress(int i, const SampleDatum *d, Stress *s) {
    float2 s0, s1, s2;
    const float2 *ss = (const float2*) d->ss;
    s0 = ss[3*i+0]; s1 = ss[3*i+1]; s2 = ss[3*i+2];

    s->s1 = make_float3(s0.x, s0.y, s1.x);
    s->s2 = make_float3(s1.y, s2.x, s2.y);
}

#undef _S_
#undef _I_
