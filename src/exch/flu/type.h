namespace exch {
namespace flu {

/*
  pp: exchanged particles 
  cc: colors of the above

  bcc: (bulk counts): cell starts in bulk coordinates
  bss: (bulk starts): cell starts in bulk coordinates
  fss: (fragment starts): cell starts in fragment coordinates

  bii: indices of the particles in bulk coordinates
*/

struct Pack {
    intp26 bcc, bss, bii, fss;
    int *counts_dev;
    int26 cap;

    comm::dBags dpp, dcc;
    comm::hBags hpp, hcc, hfss;
};

struct Comm {
    comm::Comm pp, cc, fss;
};

struct Unpack {
    comm::hBags hpp, hcc, hfss;
    comm::dBags dpp, dcc, dfss;
};

} // flu
} // exch
