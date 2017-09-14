#include <stdio.h>
#include <stdint.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "algo/scan/int.h"
#include "clist/imp.h"
#include "rnd/imp.h"

#include "int.h"
#include "imp.h"

namespace flu {

void alloc_quants(Quants *q) {
    q->n = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    Dalloc(&q->pp0, MAX_PART_NUM);
    ini(XS, YS, ZS, /**/ q->cells);
    ini_ticket(q->cells, /**/ q->tcells);
    q->pp_hst = new Particle[MAX_PART_NUM];
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp)); CC(cudaFree(q->pp0));
    fin(q->cells);
    fin_ticket(q->tcells);
    delete[] q->pp_hst;
}

void alloc_quantsI(QuantsI *q) {
    Dalloc(&q->ii, MAX_PART_NUM);
    Dalloc(&q->ii0, MAX_PART_NUM);
    q->ii_hst = new int[MAX_PART_NUM];
}

void free_quantsI(QuantsI *q) {
    CC(cudaFree(q->ii)); CC(cudaFree(q->ii0));
    delete[] q->ii_hst;
}

void alloc_ticketZ(/**/ TicketZ *t) {
    Dalloc(&t->zip0, MAX_PART_NUM);
    Dalloc(&t->zip1, MAX_PART_NUM);
}

void free_ticketZ(/**/ TicketZ *t) {
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    cudaFree(zip0);
    cudaFree(zip1);
}

void get_ticketZ(Quants q, /**/ TicketZ *t) {
    if (q.n == 0) return;
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    sub::zip(q.pp, q.n, /**/ zip0, zip1);
}

void get_ticketRND(/**/ TicketRND *t) {
    t->rnd = new rnd::KISS(0, 0, 0, 0);
}

void free_ticketRND(/**/ TicketRND *t) {
    delete t->rnd;
}

void gen_quants(Quants *q, QuantsI *qc) {
    if (multi_solvent)
        q->n = sub::genColor(q->pp, qc->ii, /*w*/ q->pp_hst, qc->ii_hst);
    else
        q->n = sub::genGrey(q->pp, /*w*/ q->pp_hst);
}

void gen_ids(const int n, QuantsI *q) {
    sub::ii_gen(n, q->ii, q->ii_hst);
}

void strt_quants(const int id, Quants *q) {
    q->n = sub::strt(id, /**/ q->pp, /* w */ q->pp_hst);
}

void strt_ii(const char *subext, const int id, QuantsI *q) {
    sub::strt_ii(subext, id, /**/ q->ii, /* w */ q->ii_hst);
}

void strt_dump(const int id, const Quants q) {
    sub::strt_dump(id, q.n, q.pp, /* w */ q.pp_hst);
}

void strt_dump_ii(const char *subext, const int id, const QuantsI q, const int n) {
    sub::strt_dump_ii(subext, id, n, q.ii, /* w */ q.ii_hst);
}

void build_cells(/**/ Quants *q) {
    clist::build(q->n, q->n, q->pp, /**/ q->pp0, q->cells, q->tcells);
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
}

} // flu
