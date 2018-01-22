enum {OUT=BLUE_COLOR, IN=RED_COLOR};

int collision_inside_1p(const float *r, const float *vv, const int4 *tt, const int nt);

/* tags: -1 if outside, i if in ith solid */
void collision_inside_hst(const Particle *pp, const int n, int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *tags);
void collision_inside_dev(const Particle *pp, const int n, int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *tags);

void collision_get_colors(const Particle *pp, int n,
                          const Particle *i_pp, const int4 *tri,
                          int nt, int nv, int nm,
                          const float3 *minext, const float3 *maxext, /**/ int *cc);
