static void cpy_H2D(const Quants *q) {
    cH2D(q->i_pp, q->i_pp_hst, q->ns * q->nv);
    cH2D(q->ss,   q->ss_hst,   q->ns);
    cH2D(q->rr0,  q->rr0_hst,  q->nps * 3);
    cH2D(q->pp,   q->pp_hst,   q->n);
}


