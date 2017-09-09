/* types local for sim:: */
namespace o {
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
