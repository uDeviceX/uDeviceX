typedef Sarray <float2*, 26> float2p26;
typedef Sarray <int*, 26> intp26;

// tag::struct[]
struct Opt {
    bool colors, ids;
};

struct DFluPack {
    DMap map;
    dBags dpp, dii, dcc;
    hBags hpp, hii, hcc;
    int nhalo; /* number of sent particles */
    int3 L; /* subdomain size */
    Opt opt;
};

struct DFluComm {
    Comm *pp, *ii, *cc;
    Opt opt;
};

struct DFluUnpack {
    hBags hpp, hii, hcc;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo; /* number of received particles */
    int3 L; /* subdomain size */
    Opt opt;
};
// end::struct[]
