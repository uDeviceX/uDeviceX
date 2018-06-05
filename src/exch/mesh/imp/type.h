enum {
    MAX_NBAGS = 1
};

struct EMeshPack {
    EMap map;
    float3 *minext, *maxext;
    dBags dbags[MAX_NBAGS], *dpp;
    hBags hbags[MAX_NBAGS], *hpp;
    int nbags;
    CommBuffer *hbuf;
    
    int3 L; /* subdomain size */
};

struct EMeshComm {
    Comm *pp;
};

struct EMeshUnpack {
    hBags hbags[MAX_NBAGS], *hpp;
    dBags dbags[MAX_NBAGS], *dpp;
    int nbags;
    CommBuffer *hbuf;
    
    int3 L; /* subdomain size */
};

/* optional structures for sending momenta back */

struct MMap { /* map for compression of Momentum (support structure only) */
    int *cc, *ss, *subids;
};

enum {
    ID_MM, ID_II,
    MAX_NMBAGS
};

struct EMeshPackM {
    MMap maps[NFRAGS];
    int *cchst, *ccdev; /* helper to collect counts */
    dBags dbags[MAX_NMBAGS], *dmm, *dii;
    hBags hbags[MAX_NMBAGS], *hmm, *hii;
    int nbags;
    CommBuffer *hbuf;
};

struct EMeshCommM {
    Comm *comm;
};

struct EMeshUnpackM {
    dBags dbags[MAX_NMBAGS], *dmm, *dii;
    hBags hbags[MAX_NMBAGS], *hmm, *hii;
    int nbags;
    CommBuffer *hbuf;
};

typedef Sarray<MMap, 26> MMap26;
