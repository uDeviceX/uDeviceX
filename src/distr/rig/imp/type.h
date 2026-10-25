#pragma once

// tag::struct[]
/* 
 ipp: particles of the mesh
 ss:  "Rigid" structures 
*/

enum {
    ID_PP,
    ID_SS,
    RIG_MAX_NBAGS
};

struct DRigPack {
    DMap map;
    dBags dbags[RIG_MAX_NBAGS], *dipp, *dss;
    hBags hbags[RIG_MAX_NBAGS], *hipp, *hss;
    CommBuffer *hbuf;
    int nbags;
    int3 L;  /* subdomain size */
};

struct DRigComm {
    Comm *comm;
};

struct DRigUnpack {
    hBags hbags[RIG_MAX_NBAGS], *hipp, *hss;
    CommBuffer *hbuf;
    int nbags;
    int3 L;  /* subdomain size */
};
// end::struct[]
