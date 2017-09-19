namespace distr {
namespace rig {

using namespace comm;

/* 
 ipp: particles of the mesh
 ss:  "Solid" structures 
*/

struct Pack {
    Map map;
    dBags dipp, dss;
    hBags hipp, dss;
};

struct Comm {
    Stamp ipp, ss;
};

struct Unpack {
    hBags hipp, hss;
};

} // rig
} // distr
