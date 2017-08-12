#include <stdint.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "common.h"
#include "msg.h"
#include "m.h"
#include "cc.h"

#include "rnd/imp.h"
#include "l/m.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "inc/mpi.h"

#include "kl.h"

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

void gather_cells(const int *start, const int *count, const int27 starts, const int26 nc,
                  const int ncells, /**/ intp26 str, intp26 cnt, intp26 cum) {
    KL(dev::count, (k_cnf(ncells)),(starts, start, count, str, cnt));
    KL(dev::scan<32>, (26, 32 * 32), (nc, cnt, /**/ cum));
}

void copy_cells(const int27 starts, const int ncells, const intp26 srccells, /**/ intp26 dstcells) {
    KL(dev::copycells, (k_cnf(ncells)), (starts, srccells, /**/ dstcells));
}
  
void pack(const int27 starts, const int nc, const Particle *pp, const intp26 str,
          const intp26 cnt, const intp26 cum, const int26 capacity, /**/ intp26 ii, Particlep26 pp0, int *bagcounts) {
    KL(dev::fill_all, ((nc + 1) / 2, 32), (starts, pp, str, cnt, cum, capacity, /**/ ii, pp0, bagcounts));
}

void pack_ii(const int27 starts, const int nc, const int *ii, const intp26 str, const intp26 cnt, const intp26 cum,
             const int26 capacity, /**/ intp26 fii) {
    /* fii: fragii */
    KL(dev::fill_all_ii, ((nc + 1) / 2, 32), (starts, ii, str, cnt, cum, capacity, fii));
}

void copy_pp(const int *np, const Particlep26 dev, /**/ Particlep26 hst) {
    dSync();
    for (int i = 0; i < 26; ++i)
        if (np[i])
            CC(cudaMemcpyAsync(hst.d[i], dev.d[i], sizeof(Particle) * np[i], D2H));
}

void copy_ii(const int *np, const intp26 dev, /**/ intp26 hst) {
    dSync();
    for (int i = 0; i < 26; ++i)
        if (np[i])
            CC(cudaMemcpyAsync(hst.d[i], dev.d[i], sizeof(int) * np[i], D2H));
}

void post_send(MPI_Comm cart, const int ranks[], const int *np, const int26 nc, const intp26 cum,
               const Particlep26 pp, int btcs, int btc, int btp, /**/ Reqs *req) {
    for (int i = 0; i < 26; ++i) {
        MC(l::m::Isend(cum.d[i], nc.d[i], MPI_INT, ranks[i],
                       btcs + i, cart, req->cells + i));
        MC(l::m::Isend(&np[i], 1, MPI_INT, ranks[i],
                       btc + i, cart, req->counts + i));
        MC(l::m::Isend(pp.d[i], np[i], datatype::particle, ranks[i],
                       btp + i, cart, req->pp + i));
    }
}

void post_send_ii(MPI_Comm cart, const int ranks[], const int *np,
                  const intp26 ii, int bt, /**/ MPI_Request sreq[26]) {

    for (int i = 0; i < 26; ++i)
        MC(l::m::Isend(ii.d[i], np[i], MPI_INT, ranks[i], bt + i, cart, sreq + i));
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

void post_expected_recv_ii(MPI_Comm cart, const int ranks[], const int tags[], const int estimate[],
                           int bt, /**/ intp26 ii, MPI_Request rreq[26]) {
    for (int i = 0; i < 26; ++i)
        MC(l::m::Irecv(ii.d[i], estimate[i], MPI_INT, ranks[i], bt + tags[i], cart, rreq + i));
}

} // sub
} // dpdr
