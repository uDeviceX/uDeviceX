namespace distr {
namespace rig {

using namespace comm;

struct Pack {
    Map map;
    dBags dpp, dss;
    hBags hpp, dss;
};

struct Comm {
    Stamp pp, ss;
};

struct Unpack {
    hBags hpp, hss;
};

} // rig
} // distr
