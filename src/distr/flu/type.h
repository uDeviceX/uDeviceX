namespace distr {
namespace flu {

using namespace comm;

// tag::struct[]
struct DFluPack {
    DMap map;
    dBags dpp, dii, dcc;
    hBags hpp, hii, hcc;
    int nhalo; /* number of sent particles */
};

struct DFluComm {
    comm::Comm pp, ii, cc;
};

struct DFluUnpack {
    hBags hpp, hii, hcc;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo; /* number of received particles */
};
// end::struct[]

} // flu
} // distr
