namespace distr {
namespace rbc {

using namespace comm;

// tag::struct[]
struct DRbcPack {
    DMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;

    /* optional: ids */
    DMap hmap;
    hBags hii;
};

struct DRbcComm {
    /* optional: ids */
    comm::Comm pp, ii;
};

struct DRbcUnpack {
    hBags hpp;

    /* optional: ids */
    hBags hii;
};
// end::struct[]

} // rbc
} // distr
