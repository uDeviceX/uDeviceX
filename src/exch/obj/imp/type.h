/*
pp: particles
ss: starts (object-wise)
*/

enum {
    ID_PP, ID_SS,
    MAX_NBAGS
};

struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hbags[MAX_NBAGS], *hpp, *hss;

    int3 L; /* subdomain size */
};

struct EObjComm {
    Comm *comm;
};

struct EObjUnpack {
    dBags dpp;
    hBags hbags[MAX_NBAGS], *hpp, *hss;

    int3 L; /* subdomain size */
};

struct EObjPackF {
    dBags dff;
    hBags hff;
};

struct EObjCommF {
    Comm *comm;
};

struct EObjUnpackF {
    hBags hff;
    dBags dff;
};
