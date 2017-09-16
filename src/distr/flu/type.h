namespace distr {
namespace flu {

using namespace comm;

/* map helper structure */
struct Map {
    int *counts;  /* number of particles leaving in each fragment */
    int *starts;  /* cumulative sum of the above                  */
    int *ids[26]; /* indices of leaving particles                 */
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
