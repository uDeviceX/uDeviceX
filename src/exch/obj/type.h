namespace exch {
namespace obj {

struct ParticlesWrap {
    int n;
    const Particle *pp;
};

struct ForcesWrap {
    int n;
    Force *ff;
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
    dBags dff;
    hBags hff;
};

struct UnpackF {
    hBags hff;
    dBags dff;
};

} // obj
} // exch
