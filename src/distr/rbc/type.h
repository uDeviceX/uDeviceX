namespace distr {
namespace rbc {

using namespace comm;

// tag::struct[]
struct Pack {
    DMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;

    /* optional: ids */
    DMap hmap;
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
// end::struct[]

} // rbc
} // distr
