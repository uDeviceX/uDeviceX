namespace distr {
namespace flu {

using namespace comm;

struct Pack {
    Map map;
    dBags *dpp, *dii, *dcc;
    hBags *hpp, *hii, *hcc;
    int nbulk;
};

struct Comm {
    Stamp *pp, *ii, *cc;
};

struct Unpack {
    hBags *hpp, *hii, *hcc;
    int *iire, *ccre;
    Particle *ppre;
};

void build_map(int n, const Particle *pp, Pack *p);

void pack_pp(const Particle *pp, int n, /**/ Pack *p);
void pack_ii(const int *ii, int n, /**/ Pack *p);
void pack_cc(const int *ii, int n, /**/ Pack *p);

} // flu
} // distr
