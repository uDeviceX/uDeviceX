void alloc_quants(Quants *q) {
    q->n = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    Dalloc(&q->pp0, MAX_PART_NUM);
    ini(XS, YS, ZS, /**/ &q->cells);
    ini_ticket(&q->cells, /**/ &q->tcells);
    q->pp_hst = new Particle[MAX_PART_NUM];
}


void alloc_quantsI(QuantsI *q) {
    Dalloc(&q->ii, MAX_PART_NUM);
    Dalloc(&q->ii0, MAX_PART_NUM);
    q->ii_hst = new int[MAX_PART_NUM];
}

void alloc_ticketZ(/**/ TicketZ *t) {
    Dalloc(&t->zip0, MAX_PART_NUM);
    Dalloc(&t->zip1, MAX_PART_NUM);
}

void get_ticketRND(/**/ TicketRND *t) {
    t->rnd = new rnd::KISS(0, 0, 0, 0);
}
