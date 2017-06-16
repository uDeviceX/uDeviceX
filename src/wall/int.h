struct Quants {
    float4 *pp;
    int n;
};

struct Ticket {
    Logistic::KISS *rnd;
    Clist *cells;
    Texo<int> texstart;
    Texo<float4> texpp;
};

void alloc_quants(Quants *q) {
    q->n = 0;
    q->pp = NULL;
}

void alloc_ticket(Ticket *t) {
    t->rnd   = new Logistic::KISS;
    t->cells = new Clist(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM);
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

void create(int *n, Particle* pp, Quants *q, Ticket *t) {
    sub::create(n, pp, &q->n, &q->pp, t->cells, &t->texstart, &t->texpp);
}

void interactions(const Quants q, const Ticket t, const int type, const Particle *pp, const int n, Force *ff) {
    sub::interactions(type, pp, n, t.texstart, t.texpp, q.n, /**/ t.rnd, ff);
}
