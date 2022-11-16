// tag::struct[]
enum {
    MAX_NHBAGS = 2,
    MAX_NDBAGS = 1
};

struct DRbcPack {
    DMap map;
    float3 *minext, *maxext;
    dBags dbags[MAX_NDBAGS], *dpp;
    hBags hbags[MAX_NHBAGS], *hpp, *hii;
    int nbags;
    CommBuffer *hbuf;
    DMap hmap; /* host map for ids */

    int3 L; /* subdomain size */
};

struct DRbcComm {
    Comm *comm;
};

struct DRbcUnpack {
    hBags hbags[MAX_NHBAGS], *hpp, *hii;
    int nbags;
    CommBuffer *hbuf;
    
    int3 L; /* subdomain size */
};
// end::struct[]
