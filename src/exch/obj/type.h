struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hpp;
};

struct EObjComm {
    Comm pp, ff;
};

struct EObjUnpack {
    hBags hpp;
    dBags dpp;
};

struct EObjPackF {
    dBags dff;
    hBags hff;
};

struct EObjUnpackF {
    hBags hff;
    dBags dff;
};
