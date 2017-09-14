void alloc_quants(Quants *q) {
    q->n = 0;
    q->pp = NULL;
}

void alloc_ticket(Ticket *t) {
    t->rnd   = new rnd::KISS;
    ini(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM, /**/ &t->cells);
    ini_ticket(&t->cells, /**/ &t->tcells);
}

void free_quants(Quants *q) {
    if (q->pp) Dfree(q->pp);
    q->n = 0;
}

void free_ticket(Ticket *t) {
    delete t->rnd;
    t->texstart.destroy();
    t->texpp.destroy();
    fin(&t->cells);
    fin_ticket(&t->tcells);
}

void gen_quants(const sdf::Quants qsdf, /**/ int *n, Particle* pp, Quants *q) {
    gen_quants(qsdf.texsdf, n, pp, &q->n, &q->pp);
}

void strt_quants(Quants *q) {
    strt_quants(&q->n, &q->pp);
}

void gen_ticket(const Quants q, Ticket *t) {
    gen_ticket(q.n, q.pp, &t->cells, &t->tcells, &t->texstart, &t->texpp);
}

void strt_dump_templ(const Quants q) {
    strt_dump_templ(q.n, q.pp);
}
