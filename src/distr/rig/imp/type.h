// tag::struct[]
/* 
 ipp: particles of the mesh
 ss:  "Solid" structures 
*/

enum {
    ID_PP,
    ID_SS,
    MAX_NBAGS
};

struct DRigPack {
    DMap map;
    dBags dbags[MAX_NBAGS], *dipp, *dss;
    hBags hbags[MAX_NBAGS], *hipp, *hss;
    int3 L;  /* subdomain size */
};

struct DRigComm {
    Comm *ipp, *ss;
};

struct DRigUnpack {
    hBags hbags[MAX_NBAGS], hipp, hss;
    int3 L;  /* subdomain size */
};
// end::struct[]
