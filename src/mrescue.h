namespace mrescue
{
void ini(int n);
void fin();

void rescue_hst(int nt, int nv, const int *tt, const Particle *i_pp, const int ns, const int n,
                const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp);

void rescue_dev(int nt, int nv, const int *tt, const Particle *i_pp, const int ns, const int n,
                const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp);
}
