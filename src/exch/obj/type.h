namespace exch {
namespace obj {

struct ParticlesWrap {
    int n;
    const Particle *pp;
};

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
    dBags ff;
    hBags ff;
};

struct UnpackF {
    hBags ff;
    dBags ff;
};

} // obj
} // exch
