#define _S_ static __device__
#define _I_ static __device__

template <typename Pa>
_S_ void part(int i, const Pa *p, const Grid *g) {
    atomicAdd(g->p[RHO] + i, 1.f);
    atomicAdd(g->p[VX]  + i, p->v.x);
    atomicAdd(g->p[VY]  + i, p->v.y);
    atomicAdd(g->p[VZ]  + i, p->v.z);
}

_S_ void stress(int i, const PartS *p, const Grid *g) {
    atomicAdd(g->s[SXX] + i, p->s1.x);
    atomicAdd(g->s[SXY] + i, p->s1.y);
    atomicAdd(g->s[SXZ] + i, p->s1.z);

    atomicAdd(g->s[SYY] + i, p->s2.x);
    atomicAdd(g->s[SYZ] + i, p->s2.y);
    atomicAdd(g->s[SZZ] + i, p->s2.z);
}

_I_ void add_part(int i, const Part *p, const Grid *g) {
    part(i, p, g);
}

_I_ void add_part(int i, const PartS *p, const Grid *g) {
    part(i, p, g);
    stress(i, p, g);
}

#undef _S_
#undef _I_
