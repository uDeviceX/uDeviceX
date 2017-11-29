namespace distr {
namespace flu {

using namespace comm;

/* structure passed to the map            */
/* optional "deathlist" to kill particles */
struct PartList {
    bool kill;
    const Particle *pp;
    const int *deathlist;
};

struct Pack {
    Map map;
    dBags dpp, dii, dcc;
    hBags hpp, hii, hcc;
    int nbulk;
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

} // flu
} // distr
