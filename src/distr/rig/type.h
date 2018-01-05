namespace distr {
namespace rig {

using namespace comm;

// tag::struct[]

/* 
 ipp: particles of the mesh
 ss:  "Solid" structures 
*/

struct Pack {
    DMap map;
    dBags dipp, dss;
    hBags hipp, hss;
};

struct Comm {
    Stamp ipp, ss;
};

struct Unpack {
    hBags hipp, hss;
};
// end::struct[]

} // rig
} // distr
