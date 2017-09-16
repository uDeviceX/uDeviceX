namespace distr {
namespace rbc {

using namespace comm;

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
