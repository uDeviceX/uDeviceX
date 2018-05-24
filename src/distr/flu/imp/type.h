typedef Sarray <float2*, 26> float2p26;
typedef Sarray <int*, 26> intp26;

enum {
    MAX_NBAGS = 3
};

// tag::struct[]
struct Opt {
    bool colors, ids;
};

struct DFluPack {
    DMap map;
    dBags dbags[MAX_NBAGS], *dpp, *dii, *dcc;
    hBags hbags[MAX_NBAGS], *hpp, *hii, *hcc;
    int nbags;
    int nhalo; /* number of sent particles */
    int3 L; /* subdomain size */
    CommBuffer *hbuf;
};

struct DFluComm {
    Comm *comm;
};

struct DFluUnpack {
    hBags hbags[MAX_NBAGS], *hpp, *hii, *hcc;
    int nbags;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo; /* number of received particles */
    int3 L; /* subdomain size */
    Opt opt;
    CommBuffer *hbuf;
};
// end::struct[]
