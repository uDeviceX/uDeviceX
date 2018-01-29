typedef Sarray <float2*, 26> float2p26;
typedef Sarray <int*, 26> intp26;

// tag::struct[]
struct DFluPack {
    DMap map;
    dBags dpp, dii, dcc;
    hBags hpp, hii, hcc;
    int nhalo; /* number of sent particles */
    int3 L; /* subdomain size */
};

struct DFluComm {
    Comm *pp, *ii, *cc;
};

struct DFluUnpack {
    hBags hpp, hii, hcc;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo; /* number of received particles */
    int3 L; /* subdomain size */
};
// end::struct[]
