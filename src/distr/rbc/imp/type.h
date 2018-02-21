// tag::struct[]
struct DRbcPack {
    DMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;

    /* optional: ids */
    bool ids;
    DMap hmap;
    hBags hii;

    int3 L; /* subdomain size */
};

struct DRbcComm {
    /* optional: ids */
    Comm *pp, *ii;
    bool ids;
};

struct DRbcUnpack {
    hBags hpp;

    /* optional: ids */
    bool ids;
    hBags hii;

    int3 L; /* subdomain size */
};
// end::struct[]
