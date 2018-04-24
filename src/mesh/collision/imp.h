struct float3;
struct Particle;
struct int4;
struct Triangles;

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt);

void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri, 
                     int nv, int nm, const Particle *i_pp, 
                     const float3 *minext, const float3 *maxext,
                     int lab_in, int lab_out, /**/ int *labels);
