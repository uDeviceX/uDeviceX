static void pp2rr(const Particle *pp, const int n, float *rr) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c)
    rr[3*i + c] = pp[i].r[c];
}

static void gen_from_strt(Coords coords, const int id, int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst) {
    Particle *pp = new Particle[MAX_PART_NUM];
    restart::read_pp(coords, "rig", restart::TEMPL, pp, nps);
    pp2rr(pp, *nps, rr0_hst);
    delete[] pp;

    restart::read_ss(coords, "rig", id, ss_hst, ns);
    *n = *ns * (*nps);
}

void strt_quants(Coords coords, const int id, Quants *q) {
    gen_from_strt(coords, id, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

static void strt_dump_templ0(Coords coords, const int nps, const float *rr0_hst) {
    Particle *pp = new Particle[nps];
    rr2pp(rr0_hst, nps, pp);

    restart::write_pp(coords, "rig", restart::TEMPL, pp, nps);
    
    delete[] pp;
}

void strt_dump_templ(Coords coords, Quants q) {
    strt_dump_templ0(coords, q.nps, q.rr0_hst);
}


static void strt_dump(Coords coords, const int id, const int ns, const Solid *ss) {
    restart::write_ss(coords, "rig", id, ss, ns);
}

void strt_dump(Coords coords, const int id, const Quants q) {
    strt_dump(coords, id, q.ns, q.ss_hst);
}
