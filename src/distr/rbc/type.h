namespace distr {
namespace rbc {

using namespace comm;

struct Map {
    int counts[NBAGS]; /* number of cells leaving in each fragment */
    int   *ids[NBAGS]; /* indices of leaving cells                 */
};

struct Pack {
    Map map;
    dBags dpp;
    hBags hpp;
    int nbulk;
};

struct Comm {
    Stamp pp;
};

struct Unpack {
    hBags hpp;
};

} // rbc
} // distr
