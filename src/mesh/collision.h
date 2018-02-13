struct RigPinInfo;

int collision_inside_1p(const RigPinInfo *pi, const float *r, const float *vv, const int4 *tt, const int nt);

/* tags: -1 if outside, i if in ith mesh */
void collision_inside_hst(int spdir, const Particle *pp, const int n, int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *tags);
void collision_inside_dev(int spdir, const Particle *pp, const int n, int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *tags);

void collision_get_colors(int spdir, const Particle *pp, int n,
                          const Particle *i_pp, const int4 *tri,
                          int nt, int nv, int nm,
                          const float3 *minext, const float3 *maxext, /**/ int *cc);
