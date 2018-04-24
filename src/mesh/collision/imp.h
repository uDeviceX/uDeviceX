struct float3;
struct Particle;
struct int4;
struct Triangles;

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt);

void collision_get_colors(int n, const Particle *pp, Triangles *tri, 
                          int nv, int nm, const Particle *i_pp, 
                          const float3 *minext, const float3 *maxext, /**/ int *cc);
