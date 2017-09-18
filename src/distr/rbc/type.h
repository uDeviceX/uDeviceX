namespace distr {
namespace rbc {

using namespace comm;

struct Pack {
    Map map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;
};

struct Comm {
    Stamp pp;
};

struct Unpack {
    hBags hpp;
};

} // rbc
} // distr
