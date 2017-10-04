void free_quants(Quants *q) {
    CC(d::Free(q->pp)); CC(d::Free(q->pp0));
    fin(&q->cells);
    fin_ticket(&q->tcells);
    delete[] q->pp_hst;
}

void free_quantsI(QuantsI *q) {
    CC(d::Free(q->ii)); CC(d::Free(q->ii0));
    delete[] q->ii_hst;
}

void free_ticketZ(/**/ TicketZ *t) {
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    CC(d::Free(zip0));
    CC(d::Free(zip1));
}

void free_ticketRND(/**/ TicketRND *t) {
    delete t->rnd;
}
