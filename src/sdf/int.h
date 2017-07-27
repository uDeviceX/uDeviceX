namespace sdf {

struct Quants {
    cudaArray *arrsdf;
    tex3Dca<float> texsdf;
};

void alloc_quants(Quants *q);
void  free_quants(Quants *q);

void ini(Quants *q);
int who_stays(const Quants q, Particle *pp, int n, int nc, int nv, int *stay);

void bounce(const tex3Dca<float> texsdf, int n, /**/ Particle *pp);
}
