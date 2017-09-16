namespace distr {
namespace rbc {

void build_map(int nc, int nv, const Particle *pp, Pack *p);
void pack_pp(int nc, int nv, const Particle *pp, /**/ Pack *p);
void download(int nc, Pack *p);

} // rbc
} // distr
