/* types local for sim:: */
namespace o {

/* distribution */
struct Distr {
    distr::flu::Pack p;
    distr::flu::Comm c;
    distr::flu::Unpack u;
};

struct H { /* halo tickets : was h:: */
    dpdr::TicketCom tc;
    dpdr::TicketRnd trnd;
    dpdr::TicketShalo ts;
    dpdr::TicketRhalo tr;

    /* optional: flags */
    dpdr::TicketICom tic;
    dpdr::TicketSIhalo tsi;
    dpdr::TicketRIhalo tri;
};

} // o

namespace r {

/* distribution */
struct Distr {
    distr::rbc::Pack p;
    distr::rbc::Comm c;
    distr::rbc::Unpack u;
};

} // r

namespace s {

/* distribution */
struct Distr {
    distr::rig::Pack p;
    distr::rig::Comm c;
    distr::rig::Unpack u;
};

} // s

/* solute exchanger */
struct Sexch {
    exch::obj::Pack p;
    exch::obj::Unpack u;
    exch::obj::PackF pf;
    exch::obj::UnpackF uf;
    exch::obj::Comm c;
};

/* mesh exchanger */
struct Mexch {
    exch::mesh::Pack p;
    exch::mesh::Comm c;
    exch::mesh::Unpack u;
};

/* bounce back exchanger */
struct BBexch {
    exch::mesh::Pack p;
    exch::mesh::Comm c;
    exch::mesh::Unpack u;

    exch::mesh::PackM pm;
    exch::mesh::CommM cm;
    exch::mesh::UnpackM um;
};
