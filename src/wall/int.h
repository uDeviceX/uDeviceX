struct Quants {
    float4 *pp;
    int n;
};

struct Ticket {
    l::rnd::d::KISS *rnd;
    Clist *cells;
    Texo<int> texstart;
    Texo<float4> texpp;
};

void alloc_quants(Quants *q) {
    q->n = 0;
    q->pp = NULL;
}

void alloc_ticket(Ticket *t) {
    t->rnd   = new l::rnd::d::KISS;
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

void gen_quants(sdf::Quants qsdf, int *n, Particle* pp, Quants *q) {
    sub::gen_quants(qsdf.texsdf, n, pp, &q->n, &q->pp);
}

void strt_quants(Quants *q) {
    sub::strt_quants(&q->n, &q->pp);
}

void gen_ticket(const Quants q, Ticket *t) {
    sub::gen_ticket(q.n, q.pp, t->cells, &t->texstart, &t->texpp);
}

void interactions(const sdf::Quants qsdf, const Quants q, const Ticket t, const int type, const Particle *pp, const int n, Force *ff) {
    sub::interactions(qsdf.texsdf, type, pp, n, t.texstart, t.texpp, q.n, /**/ t.rnd, ff);
}

void strt_dump(const Quants q) {
    sub::strt_dump(q.n, q.pp);
}

/*
  Imaginary interface
  
  restart from particles
  need conversion to float4
  
  create_from_solvent(&q &t); -> see gen_quants
  create_from_file(&q &t);    -> see strt_quants

  sub::gen_ticket(q, &t);

  restart::read/write( (&) q.pp, (&) q.n);
  
 */
