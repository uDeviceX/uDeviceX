#include "int.h"
#include "imp.h"

void alloc_quants(Quants *q) {
    q->n = 0;
    mpDeviceMalloc(&q->pp); mpDeviceMalloc(&q->pp0);
    q->cells = new clist::Clist(XS, YS, ZS);
    q->pp_hst = new Particle[MAX_PART_NUM];
}

void free_quants(Quants *q) {
    CC(cudaFree(q->pp)); CC(cudaFree(q->pp0));
    delete q->cells;
    delete[] q->pp_hst;
}

void alloc_quantsI(QuantsI *q) {
    mpDeviceMalloc(&q->ii); mpDeviceMalloc(&q->ii0);
    q->ii_hst = new int[MAX_PART_NUM];
}

void free_quantsI(QuantsI *q) {
    CC(cudaFree(q->ii)); CC(cudaFree(q->ii0));
    delete[] q->ii_hst;
}

void alloc_ticketZ(/**/ TicketZ *t) {
    mpDeviceMalloc(&t->zip0);
    mpDeviceMalloc(&t->zip1);
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
    assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: implicit dependency */
    sub::dev::zip<<<(q.n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>(zip0, zip1, (float*)q.pp, q.n);
}

void get_ticketRND(/**/ TicketRND *t) {
    t->rnd = new rnd::KISS(0, 0, 0, 0);
}

void free_ticketRND(/**/ TicketRND *t) {
    delete t->rnd;
}

void gen_quants(Quants *q) {
    q->n = sub::gen(q->pp, q->pp_hst);
}

void gen_ids(const int n, QuantsI *q) {
    sub::ii_gen(n, q->ii, q->ii_hst);
}

void gen_tags0(const int n, QuantsI *q) {
    sub::tags0_gen(n, q->ii, q->ii_hst);
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
