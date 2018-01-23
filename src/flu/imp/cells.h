void flu_build_cells(/**/ FluQuants *q) {
    clist_build(q->n, q->n, q->pp, /**/ q->pp0, &q->cells, q->mcells);
    // swap
    Particle *tmp = q->pp;
    q->pp = q->pp0; q->pp0 = tmp;
}
