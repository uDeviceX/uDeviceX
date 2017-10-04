namespace flu {

struct Quants {
    Particle *pp, *pp0;    /* particles on device  */
    int       n;           /* particle number      */
    clist::Clist cells;   /* cell lists           */
    clist::Ticket tcells; /* cell lists ticket    */
    Particle *pp_hst;      /* particles on host    */
}; 

struct QuantsI {
    int *ii, *ii0; /* int data on device */
    int *ii_hst;   /* int data on host   */
};

struct TicketZ { /* zip */
    float4  *zip0;
    ushort4 *zip1;
};

struct TicketRND { /* random */
    rnd::KISS *rnd;
};

void ini(Quants *q);
void fin(Quants *q);

void ini(QuantsI *q);
void fin(QuantsI *q);

void ini(/**/ TicketZ *t);
void fin(/**/ TicketZ *t);

void ini(/**/ TicketRND *t);
void fin(/**/ TicketRND *t);

void gen_quants(Quants *q, QuantsI *qc);
void gen_ids(const int n, QuantsI *q);
void get_ticketZ(Quants q, /**/ TicketZ *t);

void strt_quants(const int id, Quants *q);
void strt_ii(const char *subext, const int id, QuantsI *q);

void strt_dump(const int id, const Quants q);
void strt_dump_ii(const char *subext, const int id, const QuantsI q, const int n);

/* build cells only from one array of particles fully contained in the domain */
/* warning: this will delete particles which are outside                      */
void build_cells(/**/ Quants *q);

} // flu
