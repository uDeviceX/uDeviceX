namespace tcells {
namespace sub {

void build_hst(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids);
void build_dev(const Mesh m, const Particle *i_pp, const int ns, /**/ int *starts, int *counts, int *ids, /*w*/ scan::Work *w);

} // sub
} // tcells
