struct Quants {
    float4 *pp;
    int n;
// };
// struct Ticket {
    Logistic::KISS *rnd;
    Clist *cells;
    Texo<int> texstart;
    Texo<float4> texpp;
};

void alloc_quants(Quants *q) {
    q->n = 0;
    q->pp = NULL;
    q->rnd   = new Logistic::KISS;
    q->cells = new Clist(XS + 2 * XWM, YS + 2 * YWM, ZS + 2 * ZWM);
}

void free_quants(Quants *q) {
    if (q->pp) CC(cudaFree(q->pp));
    delete q->cells;
    delete q->rnd;
}

void create(int *n, Particle* pp, Quants *q) {
    sub::create(n, pp, &q->n, &q->pp, q->cells, &q->texstart, &q->texpp);
}

void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff) {
    sub::interactions(type, pp, n, q.texstart, q.texpp, q.n, /**/ q.rnd, ff);
}
