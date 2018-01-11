namespace exch {
namespace obj {

struct Pack {
    Map map;
    dBags dpp;
    hBags hpp;
};

struct Comm {
    comm::Comm pp, ff;
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

} // obj
} // exch
