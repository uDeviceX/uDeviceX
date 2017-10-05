static void ini_ii(int **ii, int **ii0, int **ii_hst) {
    Dalloc(ii, MAX_PART_NUM);
    Dalloc(ii0, MAX_PART_NUM);
    *ii_hst = new int[MAX_PART_NUM];
}

void ini(Quants *q) {
    q->n = 0;
    Dalloc(&q->pp, MAX_PART_NUM);
    Dalloc(&q->pp0, MAX_PART_NUM);
    ini(XS, YS, ZS, /**/ &q->cells);
    ini_ticket(&q->cells, /**/ &q->tcells);
    q->pp_hst = new Particle[MAX_PART_NUM];

    if (global_ids)    ini_ii(&q->ii, &q->ii0, &q->ii_hst);
    if (multi_solvent) ini_ii(&q->cc, &q->cc0, &q->cc_hst);
}


void ini(/**/ TicketZ *t) {
    Dalloc(&t->zip0, MAX_PART_NUM);
    Dalloc(&t->zip1, MAX_PART_NUM);
}

void ini(/**/ TicketRND *t) {
    t->rnd = new rnd::KISS(0, 0, 0, 0);
}
