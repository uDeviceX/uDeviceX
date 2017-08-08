#include <limits> /* for rnd */
#include <stdint.h>
#include "rnd.h"

#include <mpi.h>
#include "m.h"
#include "l/m.h"
#include "inc/type.h"
#include "common.h"
#include "common.cuda.h"
#include "common.mpi.h"

#include <conf.h>
#include "k/read.h"
#include "k/common.h"

#include "dpdr/type.h"
#include "dpdr/imp.h"

#include "dpdr/dev.h"
#include "dpdr/ini.h"
#include "dpdr/fin.h"
#include "dpdr/buf.h"
#include "dpdr/recv.h"

namespace dpdr {
namespace sub {

void wait_req(MPI_Request r[26]) {
    MPI_Status ss[26];
    MC(l::m::Waitall(26, r, ss));
}

void wait_Reqs(Reqs *r) {
    wait_req(r->cells);
    wait_req(r->pp);
    wait_req(r->counts);
}

void gather_cells(const int *start, const int *count, const int27 fragstarts, const int26 fragnc,
                  const int ncells, /**/ intp26 fragstr, intp26 fragcnt, intp26 fragcum) {
    if (ncells) dev::count<<<k_cnf(ncells)>>>(fragstarts, start, count, fragstr, fragcnt);
    dev::scan<32><<<26, 32 * 32>>>(fragnc, fragcnt, /**/ fragcum);
}

void copy_cells(const int27 fragstarts, const int ncells, const intp26 srccells, /**/ intp26 dstcells) {
    if (ncells) dev::copycells<<<k_cnf(ncells)>>>(fragstarts, srccells, /**/ dstcells);
}
  
void pack(const int27 fragstarts, const int ncells, const Particle *pp, const intp26 fragstr,
          const intp26 fragcnt, const intp26 fragcum, const int26 fragcapacity, /**/ intp26 fragii, Particlep26 fragpp, int *bagcounts) {
    if (ncells)
        dev::fill_all<<<(ncells + 1) / 2, 32>>>(fragstarts, pp, fragstr, fragcnt, fragcum,
                                                fragcapacity, /**/ fragii, fragpp, bagcounts);
}

void pack_ii(const int27 starts, const int nc, const int *ii, const intp26 str, const intp26 cnt, const intp26 cum,
             const int26 capacity, /**/ intp26 fii) {
    /* fii: fragii */
    if (nc)
        dev::fill_all_ii<<<(nc + 1) / 2, 32>>>(starts, ii, str, cnt, cum, capacity, fii);
}

void copy_pp(const int *fragnp, const Particlep26 fragppdev, /**/ Particlep26 fragpphst) {
    // dSync(); /* wait for fill_all */ /* use async copy now, no need to wait */
    for (int i = 0; i < 26; ++i)
        if (fragnp[i])
            CC(cudaMemcpyAsync(fragpphst.d[i], fragppdev.d[i], sizeof(Particle) * fragnp[i], D2H));
}

void copy_ii(const int *fragnp, const intp26 fragiidev, /**/ intp26 fragiihst) {
    // dSync(); /* wait for fill_all_ii */ /* use async copy now, no need to wait */
    for (int i = 0; i < 26; ++i)
        if (fragnp[i])
            CC(cudaMemcpyAsync(fragiihst.d[i], fragiidev.d[i], sizeof(int) * fragnp[i], D2H));
}

void post_send(MPI_Comm cart, const int ranks[], const int *np, const int26 nc, const intp26 cum,
               const Particlep26 pp, int btcs, int btc, int btp, /**/ Reqs *req) {
    for (int i = 0; i < 26; ++i) {
        MC(l::m::Isend(cum.d[i], nc.d[i], MPI_INT, ranks[i],
                       btcs + i, cart, req->cells + i));
        MC(l::m::Isend(&np[i], 1, MPI_INT, ranks[i],
                       btc + i, cart, req->counts + i));
        MC(l::m::Isend(pp.d[i], np[i], datatype::particle,
                       ranks[i], btp + i, cart, req->pp + i));
    }
}

void post_send_ii(MPI_Comm cart, const int dstranks[], const int *fragnp,
                  const intp26 fragii, int bt, /**/ MPI_Request sreq[26]) {

    for (int i = 0; i < 26; ++i)
        MC(l::m::Isend(fragii.d[i], fragnp[i], MPI_INT, dstranks[i], bt + i, cart, sreq + i));
}

void post_expected_recv(MPI_Comm cart, const int ranks[], const int tags[], const int estimate[], const int26 nc,
                        int btcs, int btc, int btp, /**/
                        Particlep26 pp, int *np, intp26 cum, Reqs *rreq) {
    for (int i = 0; i < 26; ++i) {
        MC(l::m::Irecv(pp.d[i], estimate[i], datatype::particle, ranks[i],
                       btp + tags[i], cart, rreq->pp + i));
        MC(l::m::Irecv(cum.d[i], nc.d[i], MPI_INT, ranks[i],
                       btcs + tags[i], cart, rreq->cells + i));
        MC(l::m::Irecv(np + i, 1, MPI_INT, ranks[i],
                       btc + tags[i], cart, rreq->counts + i));
    }
}

void post_expected_recv_ii(MPI_Comm cart, const int dstranks[], const int recv_tags[], const int estimate[],
                           int bt, /**/ intp26 fragii, MPI_Request rreq[26]) {
    for (int i = 0; i < 26; ++i)
        MC(l::m::Irecv(fragii.d[i], estimate[i], MPI_INT, dstranks[i], bt + recv_tags[i], cart, rreq + i));
}

} // sub
} // dpdr
