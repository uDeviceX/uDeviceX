namespace tcells {
namespace sub {

void build_hst(int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids);
void build_dev(int nt, int nv, const int4 *tt, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids, /*w*/ scan::Work *w);

} // sub
} // tcells
