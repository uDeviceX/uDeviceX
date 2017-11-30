namespace sdf {
namespace sub {

void ini(MPI_Comm cart, cudaArray *arrsdf, tex3Dca<float>*);
void bulk_wall(const tex3Dca<float>, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int  who_stays(const tex3Dca<float>, Particle *pp, int n, int nc, int nv, /**/ int *stay);
void bounce(const tex3Dca<float>, int n, /**/ Particle *pp);

}
}
