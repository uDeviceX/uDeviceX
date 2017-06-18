namespace mrescue
{
void ini(int n);
void close();

void rescue_hst(const Mesh m, const Particle *i_pp, const int ns, const int n,
                const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp);

void rescue_dev(const Mesh m, const Particle *i_pp, const int ns, const int n,
                const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp);
}
