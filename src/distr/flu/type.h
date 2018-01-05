namespace distr {
namespace flu {

using namespace comm;

// tag::struct[]
struct Pack {
    Map map;
    dBags dpp, dii, dcc;
    hBags hpp, hii, hcc;
    int nhalo; /* number of sent particles */
};

struct Comm {
    Stamp pp, ii, cc;
};

struct Unpack {
    hBags hpp, hii, hcc;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo;
};
// end::struct[]

} // flu
} // distr
