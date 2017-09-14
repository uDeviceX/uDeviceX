namespace distr {
namespace flu {

using namespace comm;

void build_map(int n, const Particle *pp, Map m);
void pack_pp(const Map m, const Particle *pp, int n, /**/ dBags bags);
void pack_ii(const Map m, const int *ii, int n, /**/ dBags bags);

int unpack_pp(const hBags bags, /**/ Particle *pp);
int unpack_ii(const hBags bags, /**/ int *ii);


} // flu
} // distr
