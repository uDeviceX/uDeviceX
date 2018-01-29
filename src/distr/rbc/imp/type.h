// tag::struct[]
struct DRbcPack {
    DMap map;
    float3 *minext, *maxext;
    dBags dpp;
    hBags hpp;

    /* optional: ids */
    DMap hmap;
    hBags hii;

    int3 L; /* subdomain size */
};

struct DRbcComm {
    /* optional: ids */
    Comm *pp, *ii;
};

struct DRbcUnpack {
    hBags hpp;

    /* optional: ids */
    hBags hii;

    int3 L; /* subdomain size */
};
// end::struct[]
