#pragma once

/*
pp: particles
cc: counts of object types
*/

enum {
    EXCH_OBJ_ID_PP, EXCH_OBJ_ID_CC,
    EXCH_OBJ_MAX_NBAGS
};

struct EObjPack {
    EMap map;
    dBags dpp;
    hBags hbags[EXCH_OBJ_MAX_NBAGS], *hpp, *hcc;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EObjComm {
    Comm *comm;
};

struct EObjUnpack {
    dBags dpp;
    hBags hbags[EXCH_OBJ_MAX_NBAGS], *hpp, *hcc;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EObjPackF {
    dBags dff;
    hBags hff;
};

struct EObjCommF {
    Comm *comm;
};

struct EObjUnpackF {
    hBags hff;
    dBags dff;
};
