struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hpp;

    int3 L; /* subdomain size */
};

struct EObjComm {
    Comm *pp, *ff;
};

struct EObjUnpack {
    hBags hpp;
    dBags dpp;

    int3 L; /* subdomain size */
};

struct EObjPackF {
    dBags dff;
    hBags hff;
};

struct EObjUnpackF {
    hBags hff;
    dBags dff;
};
