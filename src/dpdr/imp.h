namespace dpdr {
namespace sub {

// imp.h
void cancel_req(MPI_Request r[26]);
void cancel_Reqs(Reqs *r);

void wait_req(MPI_Request r[26]);
void wait_Reqs(Reqs *r);

void gather_cells(const int *start, const int *count, const int27 fragstarts, const int26 fragnc,
                  const int ncells, /**/ intp26 fragstr, intp26 fragcnt, intp26 fragcum);

void copy_cells(const int27 fragstarts, const int ncells, const intp26 srccells, /**/ intp26 dstcells);
  
void pack(const int27 fragstarts, const int ncells, const Particle *pp, const intp26 fragstr,
          const intp26 fragcnt, const intp26 fragcum, const int26 fragcapacity, /**/ intp26 fragii, Particlep26 fragpp, int *bagcounts);

void pack_ii(const int27 fragstarts, const int ncells, const int *ii, const intp26 fragstr, const intp26 fragcnt, const intp26 fragcum,
             const int26 fragcapacity, /**/ intp26 fragii);

void copy_pp(const int *fragnp, const Particlep26 fragppdev, /**/ Particlep26 fragpphst);
void copy_ii(const int *fragnp, const intp26 fragiidev, /**/ intp26 fragiihst);

void post_send(MPI_Comm cart, const int dstranks[], const int *fragnp, const int26 fragnc, const intp26 fragcum,
               const Particlep26 fragpp, int btcs, int btc, int btp, /**/ Reqs *sreq);

void post_send_ii(MPI_Comm cart, const int dstranks[], const int *fragnp,
                  const intp26 fragii, int bt, /**/ MPI_Request sreq[26]);

void post_expected_recv(MPI_Comm cart, const int dstranks[], const int recv_tags[], const int estimate[], const int26 fragnc,
                        int btcs, int btc, int btp, /**/ Particlep26 fragpp, int *Rfragnp, intp26 Rfragcum, Reqs *rreq);

void post_expected_recv_ii(MPI_Comm cart, const int dstranks[], const int recv_tags[], const int estimate[],
                           int bt, /**/ intp26 fragii, MPI_Request rreq[26]);

void recv(const int *np, const int *nc, /**/ Rbufs *b);
void recv_ii(const int *np, /**/ RIbuf *b);

// ini.h
void ini_tcom(MPI_Comm cart, /**/ MPI_Comm *newcart, int dstranks[], int recv_tags[]);
void ini_trnd(const int dstranks[], /**/ rnd::KISS* interrank_trunks[], bool interrank_masks[]);
void ini_ticketSh(/**/ Sbufs *b, int26 *est, int26 *nc);
void ini_ticketRh(/**/ Rbufs *b, int26 *est, int26 *nc);
void ini_ticketSIh(/**/ SIbuf *b);
void ini_ticketRIh(/**/ RIbuf *b);
    
// fin.h
void fin_tcom(const bool first, /**/ MPI_Comm *cart, Reqs *sreq, Reqs *rreq);
void fin_trnd(/**/ rnd::KISS* interrank_trunks[]);

// buf.h
void alloc_Sbufs(const int26 estimates, const int26 nfragcells, /**/ Sbufs *b);
void free_Sbufs(/**/ Sbufs *b);

void alloc_Rbufs(const int26 estimates, const int26 nfragcells, /**/ Rbufs *b);
void free_Rbufs(/**/ Rbufs *b);

void alloc_SIbuf(const int26 estimates, /**/ SIbuf *b);
void free_SIbuf(/**/ SIbuf *b);

void alloc_RIbuf(const int26 estimates, /**/ RIbuf *b);
void free_RIbuf(/**/ RIbuf *b);
}
}
