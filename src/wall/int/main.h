namespace wall {
void alloc_quants(Quants *q) {
    q->n = 0;
    q->pp = NULL;
}

void alloc_ticket(Ticket *t) {
    t->rnd   = new rnd::KISS;
    t->cells = new clist::Clist(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM);
}

void free_quants(Quants *q) {
    if (q->pp) CC(cudaFree(q->pp));
    q->n = 0;
}

void free_ticket(Ticket *t) {
    delete t->cells;
    delete t->rnd;
    t->texstart.destroy();
    t->texpp.destroy();
}

void gen_quants(const sdf::Quants qsdf, /**/ int *n, Particle* pp, Quants *q) {
    gen_quants(qsdf.texsdf, n, pp, &q->n, &q->pp);
}

void strt_quants(Quants *q) {
    strt_quants(&q->n, &q->pp);
}

void gen_ticket(const Quants q, Ticket *t) {
    gen_ticket(q.n, q.pp, t->cells, &t->texstart, &t->texpp);
}

void interactions(const sdf::Quants qsdf, const Quants q, const Ticket t, const int type, hforces::Cloud cloud, const int n, Force *ff) {
    interactions(qsdf.texsdf, type, cloud, n, t.texstart, t.texpp, q.n, /**/ t.rnd, ff);
}

void strt_dump_templ(const Quants q) {
    strt_dump_templ(q.n, q.pp);
}

}
