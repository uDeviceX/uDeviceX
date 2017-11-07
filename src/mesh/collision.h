namespace collision {

enum {OUT=BLUE_COLOR, IN=RED_COLOR};

int inside_1p(const float *r, const float *vv, const int4 *tt, const int nt);

/* tags: -1 if outside, i if in ith solid */
void inside_hst(const Particle *pp, const int n, int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *tags);
void inside_dev(const Particle *pp, const int n, int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *tags);

void get_colors(const Particle *pp, const int n, const Texo<float2> texvert, const Texo<int4> textri, const int nt,
                const int nv, const int nm, const float3 *minext, const float3 *maxext, /**/ int *cc);
}
