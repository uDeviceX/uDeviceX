static void setup_from_strt(const int id, /**/ Particle *pp, int *nc, int *n, /*w*/ Particle *pp_hst) {
    restart::read_pp("rbc", id, pp_hst, n);
    *nc = *n / nv;
    
    if (*n) cH2D(pp, pp_hst, *n);
}

void strt_quants(const char *r_templ, const int id, Quants *q) {
    sub::setup(r_templ, /**/ q->tri_hst, q->tri, q->adj0, q->adj1);
    sub::setup_from_strt(id, /**/ q->pp, &q->nc, &q->n, /*w*/ q->pp_hst);
}

static void strt_dump(const int id, const int n, const Particle *pp, /*w*/ Particle *pp_hst) {
    if (n) cD2H(pp_hst, pp, n);

    restart::write_pp("rbc", id, pp_hst, n);
}

void strt_dump(const int id, const Quants q) {
    sub::strt_dump(id, q.n, q.pp, /*w*/ q.pp_hst);
}
