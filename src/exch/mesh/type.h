namespace exch {
namespace mesh {

struct Pack {
    EMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;
};

struct Comm {
    comm::Comm pp;
};

struct Unpack {
    hBags hpp;
    dBags dpp;
};

/* optional structures for sending momenta back */

struct MMap { /* map for compression of Momentum (support structure only) */
    int *cc, *ss, *subids;
};

struct PackM {
    MMap maps[NFRAGS];
    int *cchst, *ccdev; /* helper to collect counts */
    dBags dmm, dii;
    hBags hmm, hii;
};

struct CommM {
    comm::Comm mm, ii;
};

struct UnpackM {
    dBags dmm, dii;
    hBags hmm, hii;
};

typedef Sarray<MMap, 26> MMap26;

} // mesh
} // exch
