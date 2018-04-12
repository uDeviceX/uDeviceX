#define _S_ static __device__
#define _I_ static __device__

template <typename Pa>
_S_ void part(int i, const Pa *p, const Grid *g) {
    atomicAdd(g->d[RHO] + i, 1.f);
    atomicAdd(g->d[VX]  + i, p->v.x);
    atomicAdd(g->d[VY]  + i, p->v.y);
    atomicAdd(g->d[VZ]  + i, p->v.z);
}

_S_ void stress(int i, const PartS *p, const Grid *g) {
    atomicAdd(g->d[SXX] + i, p->s1.x);
    atomicAdd(g->d[SXY] + i, p->s1.y);
    atomicAdd(g->d[SXZ] + i, p->s1.z);

    atomicAdd(g->d[SYY] + i, p->s2.x);
    atomicAdd(g->d[SYZ] + i, p->s2.y);
    atomicAdd(g->d[SZZ] + i, p->s2.z);
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
