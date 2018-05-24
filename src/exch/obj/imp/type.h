/*
pp: particles
cc: counts of object types
*/

enum {
    ID_PP, ID_CC,
    MAX_NBAGS
};

struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hbags[MAX_NBAGS], *hpp, *hcc;

    int3 L; /* subdomain size */
};

struct EObjComm {
    Comm *comm;
};

struct EObjUnpack {
    dBags dpp;
    hBags hbags[MAX_NBAGS], *hpp, *hcc;

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
