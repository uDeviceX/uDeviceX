#define _S_ static __device__
#define _I_ static __device__

_I_ void add_part(int i, const Part *p, const Grid *g) {
    atomicAdd(g->p[RHO] + i, 1.f);
    atomicAdd(g->p[VX]  + i, p->v.x);
    atomicAdd(g->p[VY]  + i, p->v.y);
    atomicAdd(g->p[VZ]  + i, p->v.z);
}

_I_ void add_color(int i, const int c, const Grid *g) {
#define ADD_COLOR(a)                               \
    if (ENUM_COLOR(a) == c)                        \
        atomicAdd(g->c[ENUM_COLOR(a)] + i, 1.f);

    XMACRO_COLOR(ADD_COLOR)

#undef ADD_COLOR
}

_I_ void add_stress(int i, const Stress *s, const Grid *g) {
    atomicAdd(g->s[SXX] + i, s->s1.x);
    atomicAdd(g->s[SXY] + i, s->s1.y);
    atomicAdd(g->s[SXZ] + i, s->s1.z);

    atomicAdd(g->s[SYY] + i, s->s2.x);
    atomicAdd(g->s[SYZ] + i, s->s2.y);
    atomicAdd(g->s[SZZ] + i, s->s2.z);
}

#undef _S_
#undef _I_
