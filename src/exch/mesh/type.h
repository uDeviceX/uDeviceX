namespace exch {
namespace mesh {

struct EMeshPack {
    EMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;
};

struct EMeshComm {
    comm::Comm pp;
};

struct EMeshUnpack {
    hBags hpp;
    dBags dpp;
};

/* optional structures for sending momenta back */

struct MMap { /* map for compression of Momentum (support structure only) */
    int *cc, *ss, *subids;
};

struct EMeshPackM {
    MMap maps[NFRAGS];
    int *cchst, *ccdev; /* helper to collect counts */
    dBags dmm, dii;
    hBags hmm, hii;
};

struct EMeshCommM {
    comm::Comm mm, ii;
};

struct EMeshUnpackM {
    dBags dmm, dii;
    hBags hmm, hii;
};

typedef Sarray<MMap, 26> MMap26;

} // mesh
} // exch
