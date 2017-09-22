namespace exch {
namespace mesh {

struct Pack {
    Map map;
    dBags dpp;
    hBags hpp;
};

struct Comm {
    Stamp pp;
};

struct Unpack {
    hBags hpp;
    dBags dpp;
};

} // mesh
} // exch
