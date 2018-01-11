/*
  pp: exchanged particles 
  cc: colors of the above

  bcc: (bulk counts): cell starts in bulk coordinates
  bss: (bulk starts): cell starts in bulk coordinates
  fss: (fragment starts): cell starts in fragment coordinates

  bii: indices of the particles in bulk coordinates
*/

struct EFluPack {
    intp26 bcc, bss, bii, fss;
    int *counts_dev;
    int26 cap;

    dBags dpp, dcc;
    hBags hpp, hcc, hfss;
};

struct EFluComm {
    Comm *pp, *cc, *fss;
};

struct EFluUnpack {
    hBags hpp, hcc, hfss;
    dBags dpp, dcc, dfss;
};
