static void fin_ii(int *ii, int *ii0, int *ii_hst) {
    CC(d::Free(ii));
    CC(d::Free(ii0));
    delete[] ii_hst;
}

void fin(Quants *q) {
    CC(d::Free(q->pp));
    CC(d::Free(q->pp0));
    fin(&q->cells);
    fin_map(&q->mcells);
    delete[] q->pp_hst;

    if (global_ids)    fin_ii(q->ii, q->ii0, q->ii_hst);
    if (multi_solvent) fin_ii(q->cc, q->cc0, q->cc_hst);
}

void fin(/**/ TicketZ *t) {
    float4  *zip0 = t->zip0;
    ushort4 *zip1 = t->zip1;
    CC(d::Free(zip0));
    CC(d::Free(zip1));
}

void fin(/**/ TicketRND *t) {
    delete t->rnd;
}
