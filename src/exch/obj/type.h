namespace exch {
namespace obj {

struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hpp;
};

struct EObjComm {
    comm::Comm pp, ff;
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

} // obj
} // exch
