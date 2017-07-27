namespace sdf {

struct Quants {
    cudaArray *arrsdf;
    tex3Dca<float> texsdf;
};

void alloc_quants(Quants *q);
void  free_quants(Quants *q);

void ini(Quants *q);
void bulk_wall(const tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int who_stays(const Quants q, Particle *pp, int n, int nc, int nv, int *stay);

void bounce(const tex3Dca<float> texsdf, int n, /**/ Particle *pp);
}
