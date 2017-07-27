namespace flu {

struct Quants {
    Particle *pp, *pp0; /* particles on device  */
    int       n;        /* particle number      */
    clist::Clist *cells;       /* cell lists           */
    Particle *pp_hst;   /* particles on host    */
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

void alloc_quants(Quants *q);
void free_quants(Quants *q);

void alloc_quantsI(QuantsI *q);
void free_quantsI(QuantsI *q);

void alloc_ticketZ(/**/ TicketZ *t);
void free_ticketZ(/**/ TicketZ *t);
void get_ticketZ(Quants q, /**/ TicketZ *t);

void get_ticketRND(/**/ TicketRND *t);
void free_ticketRND(/**/ TicketRND *t);

void gen_quants(Quants *q);

void gen_ids(const int n, QuantsI *q);
void gen_tags0(const int n, QuantsI *q);

void strt_quants(const int id, Quants *q);
void strt_ii(const char *subext, const int id, QuantsI *q);

void strt_dump(const int id, const Quants q);
void strt_dump_ii(const char *subext, const int id, const QuantsI q, const int n);

} // flu
