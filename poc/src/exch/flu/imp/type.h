/*
  pp: exchanged particles 
  cc: colors of the above

  bcc: (bulk counts): cell counts
  bss: (bulk starts): cell starts in bulk coordinates
  fss: (fragment starts): cell starts in fragment coordinates

  bii: indices of the particles in bulk coordinates
*/

struct EFluPack {
    enum {
        MAX_NHBAGS = 3,
        MAX_NDBAGS = 2
    };

    intp26 bcc, bss, bii, fss;
    int *counts_dev;
    int26 cap;

    dBags dbags[MAX_NDBAGS], *dpp, *dcc;
    hBags hbags[MAX_NHBAGS], *hpp, *hcc, *hfss;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};

struct EFluComm {
    Comm *comm;
};

struct EFluUnpack {
    enum {
        MAX_NHBAGS = 3,
        MAX_NDBAGS = 3
    };

    dBags dbags[MAX_NDBAGS], *dpp, *dcc, *dfss;
    hBags hbags[MAX_NHBAGS], *hpp, *hcc, *hfss;
    int nbags;
    CommBuffer *hbuf;

    int3 L; /* subdomain size */
};
