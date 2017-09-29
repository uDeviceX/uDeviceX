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

/* optional structures for sending momenta back */

struct MMap { /* map for compression of Momentum */
    int *cc, *ss, *subids;
};

struct PackM {
    MMap maps[NFRAGS];
    dBags dmm;
    hBags hmm;
};

struct CommM {
    Stamp mm;
};

struct UnpackM {
    dBags dmm;
    hBags hmm;
};

} // mesh
} // exch
