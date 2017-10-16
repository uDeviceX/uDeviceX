static void gen_from_solvent(int nt, int nv, const int4 *tt, const float *vv, /* io */ Particle *opp, int *on,
                      /* o */ int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst, Particle *pp_hst) {
    // generate models
    MSG("start solid ini");
    ic::ini("rigs-ic.txt", nt, nv, tt, vv, /**/ ns, nps, rr0_hst, ss_hst, on, opp, pp_hst);
    MSG("done solid ini");

    *n = *ns * (*nps);
}

static void gen_pp_hst(const int ns, const float *rr0_hst, const int nps, /**/ Solid *ss_hst, Particle *pp_hst) {
    rig::generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    rig::reinit_ft_hst(ns, /**/ ss_hst);
}

static void gen_ipp_hst(const Solid *ss_hst, const int ns, int nv, const float *vv, Particle *i_pp_hst) {
    rig::mesh2pp_hst(ss_hst, ns, nv, vv, /**/ i_pp_hst);
}


void gen_quants(/* io */ Particle *opp, int *on, /**/ Quants *q) {
    gen_from_solvent(q->nt, q->nv, q->htt, q->hvv, /* io */ opp, on, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst, q->pp_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void set_ids(const int ns, /**/ Solid *ss_hst, Solid *ss_dev) {
    ic::set_ids(ns, /**/ ss_hst);
    if (ns) cH2D(ss_dev, ss_hst, ns);
}

void set_ids(Quants q) {
    set_ids(q.ns, q.ss_hst, q.ss);
}
