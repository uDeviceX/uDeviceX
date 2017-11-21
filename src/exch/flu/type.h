namespace exch {
namespace flu {

/*
  pp: exchanged particles 
  cc: colors of the above

  bcc: (bulk counts): cell starts in bulk coordinates
  bss: (bulk starts): cell starts in bulk coordinates
  fss: (fragment starts): cell starts in fragment coordinates
*/

struct Pack {
    intp26 bcc, bss;
    comm::dBags dpp, dcc, dfss;
    comm::hBags hpp, hcc, hfss;
};

struct Comm {
    comm::Stamp pp, cc, fss;
};

struct Unpack {
    comm::hBags hpp, hcc, hfss;
    comm::dBags dpp, dcc, dfss;
};

} // flu
} // exch
