namespace sdf {
namespace sub {

void ini(cudaArray *arrsdf, tex3Dca<float> *texsdf);
void bulk_wall(const tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int  who_stays(const tex3Dca<float> texsdf, Particle *pp, int n, int nc, int nv, /**/ int *stay);
void bounce(const tex3Dca<float> texsdf, int n, /**/ Particle *pp);

}
}
