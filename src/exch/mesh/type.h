namespace exch {
namespace mesh {

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
    dBags dpp;
};

} // mesh
} // exch
