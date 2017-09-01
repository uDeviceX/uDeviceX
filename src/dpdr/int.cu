#include <stdint.h>
#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "d/api.h"
#include "frag.h"

#include "rnd/imp.h"

#include "mpi/basetags.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "hforces/cloud/type.h"
#include "hforces/cloud/int.h"
#include "hforces/imp.h"

#include "dpdr/type.h"
#include "dpdr/int.h"
#include "dpdr/imp.h"

namespace dpdr {
void ini_ticketcom(MPI_Comm cart, /*io*/ basetags::TagGen *tg, /**/ TicketCom *t) {
    sub::ini_tcom(cart, /**/ &t->cart, t->dstranks, t->recv_tags); 
    t->first = true;
    t->btc  = get_tag(tg);
    t->btcs = get_tag(tg);
    t->btp  = get_tag(tg);
}

void free_ticketcom(/**/ TicketCom *t) {
    sub::fin_tcom(t->first, /**/ &t->cart, &t->sreq, &t->rreq);
}

void ini_ticketrnd(const TicketCom tc, /**/ TicketRnd *tr) {
    sub::ini_trnd(tc.dstranks, /**/ tr->interrank_trunks, tr->interrank_masks);
}

void free_ticketrnd(/**/ TicketRnd *tr) {
    sub::fin_trnd(/**/ tr->interrank_trunks);
}

void alloc_ticketSh(/**/ TicketShalo *t) {
    sub::ini_ticketSh(/**/ &t->b, &t->estimate, &t->nc);

    Palloc0(&t->nphst, 26);
    Link(&t->npdev, t->nphst);

    int s = t->fragstarts.d[0] = 0;
    for (int i = 0; i < 26; ++i) t->fragstarts.d[i + 1] = (s += t->nc.d[i]);
    t->ncells = s;
}

void free_ticketSh(/**/TicketShalo *t) {
    sub::free_Sbufs(/**/ &t->b);
    CC(cudaFreeHost(t->nphst));
}

void alloc_ticketRh(/**/ TicketRhalo *t) {
    sub::ini_ticketRh(/**/ &t->b, &t->estimate, &t->nc);
}

void free_ticketRh(/**/TicketRhalo *t) {
    sub::free_Rbufs(/**/ &t->b);
}

void ini_ticketIcom(/*io*/ basetags::TagGen *tg, /**/ TicketICom *t) {    
    t->first = true;
    t->bt    = get_tag(tg);    
}

void free_ticketIcom(/**/ TicketICom *t) {t->first = true;}

void alloc_ticketSIh(/**/ TicketSIhalo *t) {
    sub::ini_ticketSIh(/**/ &t->b);
}

void free_ticketSIh(/**/TicketSIhalo *t) {
    sub::free_SIbuf(/**/ &t->b);
}

void alloc_ticketRIh(/**/ TicketRIhalo *t) {
    sub::ini_ticketRIh(/**/ &t->b);
}

void free_ticketRIh(/**/TicketRIhalo *t) {
    sub::free_RIbuf(/**/ &t->b);
}


/* remote: send functions */

void gather_cells(const int *start, const int *count, /**/ TicketShalo *t) {
    sub::gather_cells(start, count, t->fragstarts, t->nc, t->ncells,
                      /**/ t->b.str, t->b.cnt, t->b.cum);
}

void copy_cells(/**/ TicketShalo *t) {
    sub::copy_cells(t->fragstarts, t->ncells, t->b.cum, /**/ t->b.cumdev);
}

void pack(const Particle *pp, /**/ TicketShalo *t) {
    sub::pack(t->fragstarts, t->ncells, pp, t->b.str, t->b.cnt, t->b.cum, t->estimate, /**/ t->b.ii, t->b.pp, t->npdev);
    sub::copy_pp(t->nphst, t->b.pp, /**/ t->b.pphst);
}

void pack_ii(const int *ii, const TicketShalo *t, /**/ TicketSIhalo *ti) {
    sub::pack_ii(t->fragstarts, t->ncells, ii, t->b.str, t->b.cnt, t->b.cum, t->estimate, /**/ ti->b.ii);
    sub::copy_ii(t->nphst, ti->b.ii, /**/ ti->b.iihst);
}

void post_send(TicketCom *tc, TicketShalo *ts) {
    if (!tc->first) sub::wait_Reqs(&tc->sreq);
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
    sub::post_send(tc->cart, tc->dstranks, ts->nphst, ts->nc, ts->b.cumhst, ts->b.pphst,
              tc->btcs, tc->btc, tc->btp, /**/ &tc->sreq);
}

void post_send_ii(const TicketCom *tc, const TicketShalo *ts, /**/ TicketICom *tic, TicketSIhalo *tsi) {
    if (!tic->first) sub::wait_req(tic->sreq);
    dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
    sub::post_send_ii(tc->cart, tc->dstranks, ts->nphst, tsi->b.iihst, tic->bt, /**/ tic->sreq);

}

void post_expected_recv(TicketCom *tc, TicketRhalo *tr) {
    sub::post_expected_recv(tc->cart, tc->dstranks, tc->recv_tags, tr->estimate.d, tr->nc,
                            tc->btcs, tc->btc, tc->btp, /**/ tr->b.pphst, tr->np.d, tr->b.cumhst, &tc->rreq);
}

void post_expected_recv_ii(const TicketCom *tc, const TicketRhalo *tr, /**/ TicketICom *tic, TicketSIhalo *tsi) {
    sub::post_expected_recv_ii(tc->cart, tc->dstranks, tc->recv_tags, tr->estimate.d, tic->bt, /**/ tsi->b.iihst, tic->rreq);
}

void wait_recv(TicketCom *tc) {
    sub::wait_Reqs(&tc->rreq);
}

void wait_recv_ii(TicketICom *tc) {
    sub::wait_req(tc->rreq);
}

void recv(TicketRhalo *t) {
    sub::recv(t->np.d, t->nc.d, /**/ &t->b);
}

void recv_ii(const TicketRhalo *t, /**/ TicketRIhalo *ti) {
    sub::recv_ii(t->np.d, /**/ &ti->b);
}


// TODO move this to imp
void fremote(TicketRnd trnd, TicketShalo ts, TicketRhalo tr, /**/ Force *ff) {
    enum {X, Y, Z};
    int i;
    int dx, dy, dz;
    int m0, m1, m2;
    hforces::CloudA clouda;
    hforces::CloudB cloudb;

    hforces::SFrag26 sfrag;
    hforces::Frag26   frag;
    hforces::Rnd26     rnd;

    for (i = 0; i < 26; ++i) {
        dx = frag_to_dir[i][X];
        dy = frag_to_dir[i][Y];
        dz = frag_to_dir[i][Z];

        m0 = 0 == dx;
        m1 = 0 == dy;
        m2 = 0 == dz;

        hforces::ini_cloudA(ts.b.pp.d[i], &clouda);
        hforces::ini_cloudB(tr.b.pp.d[i], &cloudb);

        sfrag.d[i] = {
            (float*)ts.b.pp.d[i],
            ts.b.ii.d[i],
            clouda,            
            ts.nphst[i]};

        frag.d[i] = {
            (float2*)tr.b.pp.d[i],
            cloudb,
            tr.b.cum.d[i],
            dx,
            dy,
            dz,
            1 + m0 * (XS - 1),
            1 + m1 * (YS - 1),
            1 + m2 * (ZS - 1),
            (hforces::FragType)(abs(dx) + abs(dy) + abs(dz))};

        rnd.d[i] = {trnd.interrank_trunks[i]->get_float(), trnd.interrank_masks[i]};
    }

    hforces::interactions(sfrag, frag, rnd, (float*)ff);
}
}
