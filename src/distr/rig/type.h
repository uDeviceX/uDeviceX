namespace distr {
namespace rig {

using namespace comm;

// tag::struct[]

/* 
 ipp: particles of the mesh
 ss:  "Solid" structures 
*/

struct DRigPack {
    DMap map;
    dBags dipp, dss;
    hBags hipp, hss;
};

struct DRigComm {
    comm::Comm ipp, ss;
};

struct DRigUnpack {
    hBags hipp, hss;
};
// end::struct[]

} // rig
} // distr
