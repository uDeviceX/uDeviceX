namespace exch {
namespace bb {

struct Pack {
    Map map;
    dBags dpp;
    hBags hpp;
};

struct Comm {
    Stamp pp, ff;
};

struct Unpack {
    hBags hpp;
    dBags dpp;
};

struct PackF {
    dBags dff;
    hBags hff;
};

struct UnpackF {
    hBags hff;
    dBags dff;
};

} // bb
} // exch
