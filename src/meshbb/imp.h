namespace meshbb {

struct BBdata {
    int *ncols;       /* number of possible collisions per particle      */
    float4 *datacol;  /* list of data related to collisions per particle */
    int *idcol;       /* list of triangle colliding ids per particle     */
};

void ini(int maxpp, /**/ BBdata *d);
void fin(/**/ BBdata *d);

void reini(int n, /**/ BBdata d);
void find_collisions(int nm, int nt, int nv, const int4 *tt, const Particle *i_pp, int3 L,
                     const int *starts, const int *counts, const Particle *pp, const Force *ff, /**/ BBdata d);
void select_collisions(int n, /**/ BBdata d);
void bounce(int n, BBdata d, const Force *ff, int nt, int nv, const int4 *tt, const Particle *i_pp, /**/ Particle *pp, Momentum *mm);

void collect_momentum(int ns, int nt, int nv, const int4 *tt, const Particle *pp, const Momentum *mm, /**/ Solid *ss);
/* TODO collect for rbcs */

} // meshbb
