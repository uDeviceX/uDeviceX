namespace distr {
namespace rbc {

using namespace comm;

struct Pack {
    Map map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;

    /* optional: ids */
    Map hmap;
    hBags hii;
};

struct Comm {
    Stamp pp;

    /* optional: ids */
    Stamp ii;
};

struct Unpack {
    hBags hpp;

    /* optional: ids */
    hBags hii;
};

} // rbc
} // distr
