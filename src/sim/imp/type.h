/* types local for sim:: */
namespace o {
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

} /* namespace */
