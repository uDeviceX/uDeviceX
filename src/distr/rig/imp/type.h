// tag::struct[]
/* 
 ipp: particles of the mesh
 ss:  "Solid" structures 
*/

struct DRigPack {
    DMap map;
    dBags dipp, dss;
    hBags hipp, hss;
    int3 L;  /* subdomain size */
};

struct DRigComm {
    Comm *ipp, *ss;
};

struct DRigUnpack {
    hBags hipp, hss;
    int3 L;  /* subdomain size */
};
// end::struct[]
