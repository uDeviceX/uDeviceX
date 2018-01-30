struct EMeshPack {
    EMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;

    int3 L; /* subdomain size */
};

struct EMeshComm {
    Comm *pp;
};

struct EMeshUnpack {
    hBags hpp;
    dBags dpp;

    int3 L; /* subdomain size */
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
    Comm *mm, *ii;
};

struct EMeshUnpackM {
    dBags dmm, dii;
    hBags hmm, hii;
};

typedef Sarray<MMap, 26> MMap26;
